"""导入流程状态定义模块。"""

import copy
from typing import TypedDict, List, Dict, Any


class ImportGraphState(TypedDict, total=False):
    """导入流程在节点间传递的状态结构。"""

    task_id: str
    is_md_read_enabled: bool
    is_pdf_read_enabled: bool
    import_file_path: str
    file_dir: str
    pdf_path: str
    md_path: str
    file_title: str
    md_content: str
    chunks: list
    cache_hit: bool
    # 育儿领域新增：从目录名推断的元数据
    source_category: str       # 育儿建议/专家建议/亲子案例/沟通话术/知识科普
    source_file: str
    source_path: str


GRAPH_DEFAULT_STATE: ImportGraphState = {
    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "file_dir": "",
    "import_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "md_content": "",
    "chunks": [],
    "cache_hit": False,
    "source_category": "",
    "source_file": "",
    "source_path": "",
}


def create_default_state(**overrides) -> ImportGraphState:
    """创建一份新的默认状态，并支持局部覆盖。"""
    state = copy.deepcopy(GRAPH_DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    """获取一份干净的默认状态副本。"""
    return copy.deepcopy(GRAPH_DEFAULT_STATE)
