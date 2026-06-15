"""文件上传业务服务 - 育儿知识库版本。

支持目录扫描批量导入、单文件导入，以及文件上传到 MinIO。
"""

import os
import logging
import shutil
import time
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path
from fastapi import UploadFile
from pymilvus import MilvusClient

from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import FileProcessingError, MinioError
from knowledge.processor.import_process.main_graph import import_app
from knowledge.processor.import_process.nodes.entry_node import DirectoryEntryNode, EntryNode
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.task_util import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    add_running_task,
    add_done_task,
    add_node_duration,
    get_done_task_list,
    get_running_task_list,
    get_task_result,
    get_task_status,
    set_task_result,
    update_task_status,
)
from knowledge.utils.sse_util import create_sse_queue, push_sse_event, SSEEvent, remove_sse_queue

logger = logging.getLogger(__name__)

_MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))


class UpLoadService:
    SUPPORTED_EXTENSIONS = {".pdf", ".md"}

    def get_base_dir(self) -> str:
        return os.path.join(get_local_base_dir(), datetime.now().strftime("%Y%m%d"))

    def run_import_graph(self, task_id: str, import_file_path: str, file_dir: str):
        """运行整个导入流程。"""
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        graph_state = {
            "task_id": task_id,
            "import_file_path": import_file_path,
            "file_dir": file_dir,
        }
        try:
            for event in import_app.stream(graph_state):
                for node_name, node_state in event.items():
                    logger.info(f"当前正在执行的节点--->{node_name}")
            update_task_status(task_id, TASK_STATUS_COMPLETED)
            push_sse_event(task_id, SSEEvent.FINAL, {
                "status": TASK_STATUS_COMPLETED,
                "done_list": get_done_task_list(task_id),
                "running_list": get_running_task_list(task_id),
            })
            logger.info(f"[{task_id}] 导入流程已完成: {import_file_path}")
        except Exception as e:
            logger.error(f"[{task_id}] 执行导入过程中出现异常: {e}")
            set_task_result(task_id, "error", str(e))
            update_task_status(task_id, TASK_STATUS_FAILED)
            push_sse_event(task_id, SSEEvent.FINAL, {
                "status": TASK_STATUS_FAILED,
                "error": str(e),
            })
            self._cleanup_temp_dir(file_dir)

    def process_upload_file(self, file: UploadFile):
        """处理文件上传：存储到本地 → MinIO → 返回任务信息。"""
        self._validate_upload_file(file)

        task_id = uuid.uuid4().hex
        create_sse_queue(task_id)
        add_running_task(task_id, "upload_file")
        start_time = time.time()

        base_file_dir = self.get_base_dir()
        file_dir = os.path.join(base_file_dir, task_id)
        import_file_path = self._save_to_local(file, file_dir)

        end_time = time.time()
        add_done_task(task_id, "upload_file")
        add_node_duration(task_id, "upload_file", end_time - start_time)

        return task_id, import_file_path, file_dir

    def save_origin_file_to_minio(self, import_file_path: str, filename: str) -> None:
        """后台尽力保存原始文件到 MinIO，不阻塞前端上传响应。"""
        try:
            self._save_to_minio(import_file_path, filename)
            logger.info(f"原始文件已保存到 MinIO: {filename}")
        except Exception as e:
            logger.warning(f"原始文件保存到 MinIO 失败，不影响导入任务: {filename}, error={e}")

    def check_duplicate_by_filename(self, filename: str) -> tuple[bool, str]:
        """根据上传文件名提取标题，并检查 Milvus 中是否已有同标题文档。"""
        safe_filename = self._normalize_filename(filename)
        file_title = Path(safe_filename).stem.strip()
        if not file_title:
            raise FileProcessingError(message="无法从文件名中提取有效标题")

        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            logger.warning(f"Milvus 客户端初始化失败，跳过重复标题检查: {e}")
            return False, file_title

        return self._file_title_exists(milvus_client, file_title), file_title

    def submit_directory_import(self, source_dir: str):
        """提交目录导入任务，返回任务 ID 和扫描到的文件总数。"""
        files = DirectoryEntryNode.scan_directory(source_dir)
        if not files:
            raise FileProcessingError(message=f"目录中未找到支持的 MD/PDF 文件: {source_dir}")

        task_id = uuid.uuid4().hex
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        create_sse_queue(task_id)
        add_running_task(task_id, "directory_import")
        set_task_result(task_id, "total_files", len(files))
        set_task_result(task_id, "imported_count", 0)
        set_task_result(task_id, "skipped_count", 0)
        set_task_result(task_id, "errors", [])
        return task_id, len(files)

    def run_directory_import(self, task_id: str, source_dir: str):
        """后台执行目录导入任务。"""
        start_time = time.time()
        imported_count = 0
        skipped_count = 0
        errors = []

        try:
            files = DirectoryEntryNode.scan_directory(source_dir)
            total_files = len(files)
            set_task_result(task_id, "total_files", total_files)
            work_base_dir = os.path.join(self.get_base_dir(), task_id, "directory_import")
            os.makedirs(work_base_dir, exist_ok=True)

            for index, file_path in enumerate(files, start=1):
                set_task_result(task_id, "current_file", str(file_path))
                push_sse_event(task_id, SSEEvent.PROGRESS, {
                    "status": TASK_STATUS_PROCESSING,
                    "total_files": total_files,
                    "imported_count": imported_count,
                    "skipped_count": skipped_count,
                    "current_file": str(file_path),
                    "current_index": index,
                })
                try:
                    import_file_path, file_dir = self._prepare_directory_import_file(
                        source_file_path=file_path,
                        work_base_dir=work_base_dir,
                        index=index,
                    )
                    graph_state = {
                        "task_id": task_id,
                        "import_file_path": import_file_path,
                        "file_dir": file_dir,
                        "source_file": file_path.name,
                        "source_path": str(file_path.resolve()),
                        "source_category": EntryNode._extract_category(file_path),
                    }

                    for event in import_app.stream(graph_state):
                        for node_name, _ in event.items():
                            logger.info(f"[{file_path.name}] 节点: {node_name}")

                    imported_count += 1
                    set_task_result(task_id, "imported_count", imported_count)
                except Exception as e:
                    skipped_count += 1
                    error_text = f"{file_path.name}: {str(e)}"
                    errors.append(error_text)
                    set_task_result(task_id, "skipped_count", skipped_count)
                    set_task_result(task_id, "errors", list(errors))
                    logger.error(f"导入失败 {file_path.name}: {e}", exc_info=True)

            if imported_count > 0:
                update_task_status(task_id, TASK_STATUS_COMPLETED)
            else:
                set_task_result(task_id, "error", "目录内文件均导入失败")
                update_task_status(task_id, TASK_STATUS_FAILED)
        except Exception as e:
            set_task_result(task_id, "error", str(e))
            update_task_status(task_id, TASK_STATUS_FAILED)
            logger.error(f"目录导入任务失败: {e}", exc_info=True)
        finally:
            add_done_task(task_id, "directory_import")
            add_node_duration(task_id, "directory_import", time.time() - start_time)
            set_task_result(task_id, "current_file", "")
            push_sse_event(task_id, SSEEvent.FINAL, {
                "status": get_task_status(task_id),
                "total_files": get_task_result(task_id, "total_files", 0),
                "imported_count": get_task_result(task_id, "imported_count", 0),
                "skipped_count": get_task_result(task_id, "skipped_count", 0),
                "errors": get_task_result(task_id, "errors", []),
            })

    def process_directory_import(self, source_dir: str):
        """同步处理目录导入，保留给本地脚本或兼容调用使用。"""
        task_id, total = self.submit_directory_import(source_dir)
        self.run_directory_import(task_id, source_dir)
        # 从任务结果读取真实计数，避免同步兼容方法误报。
        from knowledge.utils.task_util import get_task_result

        imported_count = get_task_result(task_id, "imported_count", 0)
        skipped_count = get_task_result(task_id, "skipped_count", 0)
        errors = get_task_result(task_id, "errors", [])

        return task_id, total, imported_count, skipped_count, errors

    def _validate_upload_file(self, file: UploadFile) -> None:
        filename = self._normalize_filename(file.filename)
        suffix = os.path.splitext(filename)[1].lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(self.SUPPORTED_EXTENSIONS))
            raise FileProcessingError(message=f"不支持的文件类型：{suffix or '无扩展名'}，仅支持：{supported}")

        try:
            file.file.seek(0, os.SEEK_END)
            file_size = file.file.tell()
            file.file.seek(0)
        except Exception:
            file_size = 0
        if file_size > _MAX_UPLOAD_BYTES:
            max_mb = _MAX_UPLOAD_BYTES / (1024 * 1024)
            actual_mb = file_size / (1024 * 1024)
            raise FileProcessingError(message=f"文件过大 ({actual_mb:.1f}MB)，最大允许 {max_mb:.0f}MB")

    def _save_to_local(self, file: UploadFile, file_dir: str) -> str:
        os.makedirs(file_dir, exist_ok=True)
        safe_filename = self._normalize_filename(file.filename)
        import_file_path = os.path.join(file_dir, safe_filename)
        try:
            with open(import_file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        except IOError as e:
            logger.error(f"{file.filename} 写入临时目录失败: {e}")
            raise FileProcessingError(message=f"{file.filename} 写入临时目录失败: {e}")
        logger.info(f"上传文件已写入本地临时目录: {import_file_path}")
        return import_file_path

    def _prepare_directory_import_file(self, source_file_path: Path, work_base_dir: str, index: int) -> tuple[str, str]:
        """把目录导入源文件复制到任务临时目录，避免污染原始数据目录。"""
        safe_filename = self._normalize_filename(source_file_path.name)
        safe_stem = Path(safe_filename).stem[:80] or f"file_{index}"
        file_dir = os.path.join(work_base_dir, f"{index:04d}_{safe_stem}_{uuid.uuid4().hex[:8]}")
        os.makedirs(file_dir, exist_ok=True)

        import_file_path = os.path.join(file_dir, safe_filename)
        try:
            shutil.copy2(source_file_path, import_file_path)
        except IOError as e:
            raise FileProcessingError(message=f"复制目录导入文件失败: {source_file_path} -> {import_file_path}: {e}")
        return import_file_path, file_dir

    def _normalize_filename(self, filename: str) -> str:
        """标准化并安全校验上传文件名，避免路径穿越和异常字符。"""
        normalized = (filename or "").strip()
        if not normalized:
            raise FileProcessingError(message="上传文件名不能为空")

        normalized = normalized.replace("\x00", "")
        normalized = unicodedata.normalize("NFC", normalized)

        safe_chars: list[str] = []
        for ch in normalized:
            if (
                "一" <= ch <= "鿿"
                or "　" <= ch <= "〿"
                or "＀" <= ch <= "￯"
                or (ch.isascii() and (ch.isalnum() or ch in "._- "))
            ):
                safe_chars.append(ch)
            else:
                safe_chars.append("_")
        normalized = "".join(safe_chars).strip()

        safe_filename = os.path.basename(normalized)
        if safe_filename != normalized:
            raise FileProcessingError(message="上传文件名不合法")
        return safe_filename

    def _file_title_exists(self, milvus_client: MilvusClient, file_title: str) -> bool:
        """检查 chunks_collection 中是否已存在同标题文档。"""
        collection_name = self._get_chunks_collection_name()
        if not collection_name:
            logger.warning("未配置 chunks_collection，跳过重复标题检查")
            return False
        if not milvus_client.has_collection(collection_name):
            logger.info(f"集合 {collection_name} 不存在，视为暂无历史导入记录")
            return False

        escaped_file_title = file_title.replace("\\", "\\\\").replace('"', '\\"')
        existing = milvus_client.query(
            collection_name=collection_name,
            filter=f'file_title == "{escaped_file_title}"',
            output_fields=["file_title"],
            limit=1,
        )
        return bool(existing)

    def _get_chunks_collection_name(self) -> str:
        """获取当前导入流程使用的 chunks 集合名。"""
        return get_config().chunks_collection

    def _save_to_minio(self, import_file_path: str, filename: str):
        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e:
            raise MinioError(message=f"MinIO 客户端获取失败: {e}", cause=e)

        bucket_name = os.getenv("MINIO_BUCKET_NAME", "parenting-kb")
        object_name = f"origin_files/{datetime.now().strftime('%Y%m%d')}/{filename}"
        try:
            minio_client.fput_object(bucket_name, object_name, import_file_path)
        except Exception as e:
            raise MinioError(message=f"文件上传到 MinIO 失败: {e}", cause=e)

    def _cleanup_temp_dir(self, file_dir: str) -> None:
        if not file_dir or not os.path.exists(file_dir):
            return
        try:
            shutil.rmtree(file_dir, ignore_errors=True)
            logger.info(f"临时目录已清理: {file_dir}")
        except Exception as e:
            logger.warning(f"临时目录清理失败 ({file_dir}): {e}")
