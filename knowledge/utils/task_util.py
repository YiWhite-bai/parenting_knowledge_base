from typing import Any, Dict, List
from collections import defaultdict
import threading

"""
任务id: 主要追踪上传文件（任务）的状态流程
上传一个文件就属于一个任务（唯一的任务id）
"""
_tasks_running_list: Dict[str, List[str]] = defaultdict(list)
_tasks_done_list: Dict[str, List[str]] = defaultdict(list)
_tasks_duration: Dict[str, Dict[str, float]] = defaultdict(dict)

_tasks_result: Dict[str, Dict[str, Any]] = defaultdict(dict)
_tasks_status: Dict[str, str] = {}

_lock = threading.Lock()

TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

_NODE_NAME_TO_CN: Dict[str, str] = {
    "upload_file": "上传文件",
    "directory_import": "批量导入",
    "entry_node": "检查文件",
    "pdf_to_md_node": "PDF转Markdown",
    "parenting_metadata_node": "提取育儿元数据",
    "document_split_node": "文档切分",
    "embedding_chunks_node": "向量生成",
    "import_milvus_node": "导入向量数据库",
    "__end__": "处理完成",
    # --- Query 流程节点 ---
    "search_router_node": "分析问题意图",
    "answer_output_node": "生成答案",
    "reranker_node": "重排序",
    "rrf_node": "倒排融合",
    "vector_search_node": "切片搜索",
    "hyde_search_node": "切片搜索(假设性文档)",
}


def _to_cn(node_name: str) -> str:
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def add_running_task(task_id: str, node_name: str) -> None:
    with _lock:
        running = _tasks_running_list[task_id]
        if node_name not in running:
            running.append(node_name)


def remove_running_task(task_id: str, node_name: str) -> None:
    with _lock:
        if node_name in _tasks_running_list[task_id]:
            _tasks_running_list[task_id].remove(node_name)


def add_done_task(task_id: str, node_name: str) -> None:
    with _lock:
        if node_name in _tasks_running_list[task_id]:
            _tasks_running_list[task_id].remove(node_name)
        done = _tasks_done_list[task_id]
        if node_name not in done:
            done.append(node_name)


def get_running_task_list(task_id: str) -> List[str]:
    with _lock:
        return [_to_cn(n) for n in _tasks_running_list.get(task_id, [])]


def get_done_task_list(task_id: str) -> List[str]:
    with _lock:
        return [_to_cn(n) for n in _tasks_done_list.get(task_id, [])]


def get_task_status(task_id: str) -> str:
    with _lock:
        return _tasks_status.get(task_id, "")


def update_task_status(task_id: str, status_name: str) -> None:
    with _lock:
        _tasks_status[task_id] = status_name


def set_task_result(task_id: str, key: str, value: Any) -> None:
    with _lock:
        _tasks_result[task_id][key] = value


def get_task_result(task_id: str, key: str, default: Any = "") -> Any:
    with _lock:
        return _tasks_result.get(task_id, {}).get(key, default)


def add_node_duration(task_id: str, node_name: str, duration: float) -> None:
    cn_name = _to_cn(node_name)
    with _lock:
        _tasks_duration[task_id][cn_name] = round(duration, 2)


def get_node_durations(task_id: str) -> Dict[str, float]:
    with _lock:
        return dict(_tasks_duration.get(task_id, {}))


def get_task_info(task_id: str) -> Dict[str, Any]:
    with _lock:
        result = _tasks_result.get(task_id, {})
        return {
            "status": _tasks_status.get(task_id, ""),
            "running_list": [_to_cn(n) for n in _tasks_running_list.get(task_id, [])],
            "done_list": [_to_cn(n) for n in _tasks_done_list.get(task_id, [])],
            "durations": dict(_tasks_duration.get(task_id, {})),
            "error": result.get("error", ""),
            "total_files": result.get("total_files", 0),
            "imported_count": result.get("imported_count", 0),
            "skipped_count": result.get("skipped_count", 0),
            "current_file": result.get("current_file", ""),
            "errors": result.get("errors", []),
        }


def clear_task(task_id: str):
    with _lock:
        _tasks_running_list.pop(task_id, None)
        _tasks_done_list.pop(task_id, None)
        _tasks_duration.pop(task_id, None)
        _tasks_status.pop(task_id, None)
        _tasks_result.pop(task_id, None)
