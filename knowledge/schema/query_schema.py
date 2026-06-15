"""查询相关 Schema 定义"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="查询内容，1-2000 字符",
    )
    session_id: Optional[str] = Field(None, description="会话ID，不传则自动生成")
    is_stream: bool = Field(False, description="是否流式返回")


class QueryResponse(BaseModel):
    message: str = Field(..., description="响应消息")
    session_id: str = Field(..., description="会话ID")
    answer: str = Field("", description="生成的答案")
    done_list: List[str] = Field(default_factory=list, description="已完成节点列表")
    running_list: List[str] = Field(default_factory=list, description="正在运行节点列表")
    status: str = Field("", description="任务状态")
    error: str = Field("", description="失败原因")
    image_urls: List[str] = Field(default_factory=list, description="答案关联图片链接")
    reranked_docs: List[Dict[str, Any]] = Field(default_factory=list, description="精排后的检索文档列表")


class StreamSubmitResponse(BaseModel):
    message: str = Field(..., description="响应消息")
    session_id: str = Field(..., description="会话ID")
    task_id: str = Field(..., description="任务ID，前端用此 ID 建立 SSE 连接")


class HistoryItem(BaseModel):
    id: str = Field("", alias="_id")
    session_id: str = ""
    role: str = ""
    text: str = ""
    rewritten_query: str = ""
    age_group: str = ""
    problem_type: str = ""
    query_domain: str = ""
    image_urls: List[str] = Field(default_factory=list)
    ts: Optional[float] = None


class HistoryResponse(BaseModel):
    session_id: str
    items: List[HistoryItem]


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="检索关键词或问题",
    )
    age_group: Optional[str] = Field(None, description="年龄段：0-3岁/3-6岁/6-12岁/12+岁")
    problem_type: Optional[str] = Field(None, description="问题类型，如情绪管理/行为引导/亲子沟通等")
    scene: Optional[str] = Field(None, description="场景关键词，如睡前拖延/入园分离/餐桌冲突")
    content_type: Optional[str] = Field(None, description="内容类型：育儿建议/专家文章/亲子案例/沟通话术/知识科普")
    top_k: int = Field(5, ge=1, le=20, description="返回结果数量")


class KnowledgeSearchResult(BaseModel):
    chunk_id: Optional[int | str] = Field(None, description="Milvus chunk 主键")
    content: str = Field("", description="内容片段")
    title: str = Field("", description="切片标题")
    file_title: str = Field("", description="文章标题")
    author: str = Field("", description="作者")
    age_group: str = Field("", description="年龄段")
    content_type: str = Field("", description="内容类型")
    problem_type: str = Field("", description="问题类型")
    scene: str = Field("", description="场景描述")
    source_file: str = Field("", description="来源文件名")
    source_path: str = Field("", description="来源路径或资源链接")
    score: float = Field(0.0, description="检索相关性分数")


class KnowledgeSearchResponse(BaseModel):
    message: str = Field(..., description="响应消息")
    query: str = Field(..., description="原始检索请求")
    total: int = Field(0, description="返回结果数量")
    items: List[KnowledgeSearchResult] = Field(default_factory=list, description="检索结果列表")
