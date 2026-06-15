"""查询流程状态类型定义。育儿版本：去掉商品相关字段，新增育儿元数据字段。"""

from typing import TypedDict, List, Dict, Annotated
import copy
import operator


class QueryGraphState(TypedDict):
    """查询流程状态"""
    session_id: str
    task_id: str
    original_query: str
    rewritten_query: str
    embedding_chunks: list  # 向量检索结果
    hyde_embedding_chunks: list  # HyDE 检索结果
    rrf_chunks: list  # RRF 融合后的切片
    reranked_docs: list  # 精排后的文档
    prompt: str  # 提示词
    answer: str  # 答案
    age_group: str  # 识别的年龄段
    problem_type: str  # 识别的问题类型
    query_domain: str  # 问题领域: parenting / general / out_of_domain
    history: list  # 历史对话
    is_stream: bool  # 是否流式输出
    image_urls: List[str]  # 答案关联图片链接
    search_errors: Annotated[List[Dict[str, str]], operator.add]  # 搜索错误（并行累加）
    route_action: str  # 路由决策: "retrieve" / "answer"
    retrieval_strategy: str  # 检索策略


DEFAULT_STATE: QueryGraphState = {
    "session_id": "",
    "task_id": "",
    "original_query": "",
    "rewritten_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "age_group": "",
    "problem_type": "",
    "query_domain": "parenting",
    "history": [],
    "is_stream": False,
    "image_urls": [],
    "search_errors": [],
    "route_action": "retrieve",
    "retrieval_strategy": "generic_local",
}


def create_default_state(**overrides) -> QueryGraphState:
    """创建默认状态，支持字段覆盖。"""
    state = copy.deepcopy(DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> QueryGraphState:
    """获取默认状态副本。"""
    return copy.deepcopy(DEFAULT_STATE)
