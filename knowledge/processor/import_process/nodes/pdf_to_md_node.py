"""PDF 转 Markdown 节点。"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import StateFieldError, PdfConversionError


class PdfToMdNode(BaseNode):
    """将 PDF 文件解析为 Markdown 的导入节点。"""

    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.logger.info(f"[{self.name}] 开始处理 PDF 转 Markdown...")

        import_file_path_obj, file_dir_obj = self._validate_state(state)
        md_path = self.get_md_path(import_file_path_obj, file_dir_obj)

        self.log_step("Step 3", "检测本地缓存文件")
        if self._is_valid_local_cache(md_path, import_file_path_obj):
            self.logger.info(f"文件已缓存，跳过 MinerU 解析，直接使用：{md_path}")
            state["md_path"] = md_path
            state["cache_hit"] = True
            return state

        self.log_step("Step 4", "执行 MinerU 解析 PDF")
        processed_code = self._execute_mineru_parse(import_file_path_obj, file_dir_obj)
        if processed_code != 0:
            raise PdfConversionError(message="MinerU解析PDF失败", node_name=self.name)

        if not Path(md_path).exists():
            raise PdfConversionError(
                message=f"MinerU返回成功，但未找到生成的MD文件: {md_path}",
                node_name=self.name,
            )
        if os.path.getsize(md_path) == 0:
            raise PdfConversionError(
                message=f"MinerU生成的MD文件为空: {md_path}",
                node_name=self.name,
            )

        state["md_path"] = md_path
        state["cache_hit"] = False
        return state

    def _validate_state(self, state: ImportGraphState) -> Tuple[Path, Path]:
        self.log_step("Step 1", "校验和获取解析文件路径和输出目录")
        import_file_path = state.get("import_file_path", "")
        if not import_file_path:
            raise StateFieldError(node_name=self.name, field_name="import_file_path", expected_type=str)

        import_file_path_obj = Path(import_file_path)
        if not import_file_path_obj.exists():
            raise StateFieldError(
                node_name=self.name, field_name="import_file_path", expected_type=str,
                message="解析文件的路径不存在"
            )

        file_dir = state.get("file_dir", "")
        if not file_dir:
            file_dir = str(import_file_path_obj.parent)
            self.logger.info(f"未指定输出目录，使用默认目录：{file_dir}")

        file_dir_obj = Path(file_dir)
        file_dir_obj.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"解析的文件路径：{import_file_path_obj}")
        self.logger.info(f"输出的文件目录：{file_dir_obj}")
        return import_file_path_obj, file_dir_obj

    def _is_valid_local_cache(self, md_path: str, pdf_path: Path) -> bool:
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            return False
        images_dir = md_path_obj.parent / "images"
        if not images_dir.exists():
            return False
        pdf_mtime = pdf_path.stat().st_mtime
        md_mtime = md_path_obj.stat().st_mtime
        if pdf_mtime > md_mtime:
            self.logger.warning("PDF 文件已被修改，历史缓存失效，将重新转换。")
            return False
        return True

    def _execute_mineru_parse(self, import_file_path_obj: Path, file_dir_obj: Path) -> int:
        cmd = [
            "mineru",
            "-p", str(import_file_path_obj),
            "-o", str(file_dir_obj),
            "--backend", "pipeline",
        ]
        self.logger.info(f"执行终端命令：{' '.join(cmd)}")
        start_time = time.time()

        proc = subprocess.Popen(
            args=cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            encoding="utf-8",
            bufsize=1,
        )

        for line in proc.stdout:
            line = line.strip()
            if line:
                self.logger.info(f"[MinerU] {line}")

        processed_result = proc.wait()
        end_time = time.time()
        if processed_result == 0:
            self.logger.info(f"MinerU 解析 PDF 成功，耗时: {end_time - start_time:.2f}s")
        else:
            self.logger.error(f"MinerU 解析 PDF 失败，退出码: {processed_result}")
        return processed_result

    def get_md_path(self, import_file_path_obj: Path, file_dir_obj: Path) -> str:
        file_name = import_file_path_obj.stem
        return str(file_dir_obj / file_name / "auto" / f"{file_name}.md")
