"""导入流程入口节点。

职责：支持目录递归扫描和单文件处理。识别文件类型并设置路由标志。
"""

from pathlib import Path
from typing import List

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import (
    ValidationError,
    FileProcessingError,
)


class EntryNode(BaseNode):
    """导入流程入口节点。"""

    name = "entry_node"
    SUPPORTED_EXTENSIONS: List[str] = [".pdf", ".md"]

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("Step 1", "获取并校验文件路径参数")
        file_dir = state.get("file_dir")
        import_file_path = state.get("import_file_path")
        if not file_dir or not import_file_path:
            raise ValidationError(
                f"文件目录或文件路径为空。file_dir={file_dir}, import_file_path={import_file_path}",
                self.name,
            )

        self.log_step("Step 2", "创建 Path 对象并验证文件")
        path = Path(import_file_path)
        if not path.exists():
            raise FileProcessingError(f"文件不存在：{import_file_path}", self.name)
        if not path.is_file():
            raise FileProcessingError(f"路径不是文件：{import_file_path}", self.name)

        self.log_step("Step 3", "检测文件类型")
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValidationError(
                f"不支持的文件类型：{suffix}。支持的类型：{', '.join(self.SUPPORTED_EXTENSIONS)}",
                self.name,
            )

        state["is_pdf_read_enabled"] = False
        state["is_md_read_enabled"] = False
        state["pdf_path"] = ""
        state["md_path"] = ""

        if suffix == ".pdf":
            self.logger.info(f"检测到 PDF 文件：{path.name}")
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = str(import_file_path)
        elif suffix == ".md":
            self.logger.info(f"检测到 Markdown 文件：{path.name}")
            state["is_md_read_enabled"] = True
            state["md_path"] = str(import_file_path)

        self.log_step("Step 4", "提取文件标题和目录分类")
        file_title = path.stem
        state["file_title"] = file_title
        state["source_file"] = state.get("source_file") or path.name
        state["source_path"] = state.get("source_path") or str(path.resolve())

        # 从父目录名推断 source_category
        source_category = state.get("source_category") or self._extract_category(path)
        state["source_category"] = source_category

        self.logger.info(f"文件标题：{file_title}, 分类目录：{source_category}")
        return state

    @staticmethod
    def _extract_category(file_path: Path) -> str:
        """从文件路径中提取育儿内容分类。"""
        CATEGORY_NAMES = {"育儿建议", "专家建议", "亲子案例", "沟通话术", "知识科普"}
        for parent in file_path.parents:
            if parent.name in CATEGORY_NAMES:
                return parent.name
        return "知识科普"


class DirectoryEntryNode(BaseNode):
    """目录级别入口节点，扫描整个目录获取所有可导入文件。"""

    name = "entry_node"
    SUPPORTED_EXTENSIONS: List[str] = [".pdf", ".md"]

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """不实际处理单个文件，只验证目录存在。批量导入由 service 层逐个调用 EntryNode。"""
        file_dir = state.get("file_dir")
        if not file_dir:
            raise ValidationError("文件目录为空", self.name)

        dir_path = Path(file_dir)
        if not dir_path.exists():
            raise FileProcessingError(f"目录不存在：{file_dir}", self.name)

        self.logger.info(f"目录验证通过：{file_dir}")
        return state

    @staticmethod
    def scan_directory(dir_path: str) -> List[Path]:
        """递归扫描目录，返回所有支持的 MD/PDF 文件。"""
        path = Path(dir_path)
        if not path.exists():
            raise FileProcessingError(f"目录不存在：{dir_path}")

        if path.is_file():
            suffix = path.suffix.lower()
            return [path] if suffix in DirectoryEntryNode.SUPPORTED_EXTENSIONS else []

        files = []
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in DirectoryEntryNode.SUPPORTED_EXTENSIONS:
                files.append(file_path)
        return files
