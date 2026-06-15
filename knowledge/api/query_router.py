"""查询服务 FastAPI 应用"""

import asyncio
import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Union, List

import uvicorn
from fastapi import FastAPI, Depends, BackgroundTasks, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from knowledge.schema.query_schema import (
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    QueryRequest,
    StreamSubmitResponse,
    QueryResponse,
)
from knowledge.schema.upload_schema import TaskStatusResponse
from knowledge.core.deps import get_query_service
from knowledge.core.paths import get_front_page_dir
from knowledge.service.query_service import QueryService
from knowledge.utils.sse_util import create_sse_queue, sse_generator
from knowledge.utils.client.storage_clients import StorageClients

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_QUERY_RATE_LIMIT = int(os.getenv("QUERY_RATE_PER_MINUTE", "20"))
_rate_window: defaultdict[str, List[float]] = defaultdict(list)


def _check_rate_limit(client_ip: str) -> None:
    now = time.monotonic()
    window_start = now - 60.0
    _rate_window[client_ip] = [t for t in _rate_window[client_ip] if t > window_start]
    if len(_rate_window[client_ip]) >= _QUERY_RATE_LIMIT:
        raise HTTPException(status_code=429, detail=f"请求过于频繁，每分钟最多 {_QUERY_RATE_LIMIT} 次")
    _rate_window[client_ip].append(now)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    yield
    StorageClients.shutdown()


def create_app():
    """创建 FastAPI 实例。"""
    app = FastAPI(
        description="育儿知识库查询服务",
        version="v1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 挂载前端静态页面
    page_dir = get_front_page_dir()
    if page_dir and os.path.exists(page_dir):
        app.mount("/front", StaticFiles(directory=page_dir), name="front")

    register_router(app)
    return app


def register_router(app: FastAPI):
    @app.get("/")
    def hello_world():
        return {"flag": "success", "service": "育儿知识库查询服务"}

    @app.post("/query")
    async def query(fastapi_request: Request,
                    request: QueryRequest,
                    background_tasks: BackgroundTasks,
                    service: QueryService = Depends(get_query_service),
                    ) -> Union[StreamSubmitResponse, QueryResponse]:
        """处理育儿查询请求"""
        _check_rate_limit(fastapi_request.client.host if fastapi_request.client else "unknown")

        session_id = request.session_id or service.generate_session_id()
        task_id = service.generate_task_id()

        if request.is_stream:
            service.submit_query(task_id=task_id, is_stream=request.is_stream)
            background_tasks.add_task(service.run_query_graph,
                                      session_id=session_id,
                                      task_id=task_id,
                                      query=request.query,
                                      is_stream=request.is_stream)
            return StreamSubmitResponse(message="查询请求已经提交", session_id=session_id, task_id=task_id)
        else:
            service.submit_query(task_id=task_id, is_stream=request.is_stream)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, service.run_query_graph, session_id, task_id, request.query,
                                       request.is_stream)
            task_payload = service.get_task_payload(task_id)
            return QueryResponse(
                message="查询请求已经处理完了",
                session_id=session_id,
                answer=task_payload.get("answer", ""),
                done_list=task_payload.get("done_list", []),
                running_list=task_payload.get("running_list", []),
                status=task_payload.get("status", ""),
                error=task_payload.get("error", ""),
                image_urls=task_payload.get("image_urls", []),
                reranked_docs=task_payload.get("reranked_docs", []),
            )

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request) -> StreamingResponse:
        """SSE 流式输出"""
        return StreamingResponse(content=sse_generator(task_id, request), media_type="text/event-stream")

    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    async def get_task_status_endpoint(
            task_id: str,
            service: QueryService = Depends(get_query_service),
    ):
        """查询任务状态"""
        task_payload = service.get_task_payload(task_id)
        return TaskStatusResponse(**task_payload)

    @app.get("/history/{session_id}")
    async def get_history(
            session_id: str, limit: int = 50,
            service: QueryService = Depends(get_query_service),
    ):
        """获取对话历史"""
        try:
            items = service.get_history(session_id, limit)
            return {"session_id": session_id, "items": items}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"history error: {e}")

    @app.post("/knowledge/search", response_model=KnowledgeSearchResponse)
    async def search_knowledge(
            request: KnowledgeSearchRequest,
            service: QueryService = Depends(get_query_service),
    ):
        """按元数据检索育儿知识片段。"""
        try:
            items = service.search_knowledge(
                query=request.query,
                age_group=request.age_group or "",
                problem_type=request.problem_type or "",
                scene=request.scene or "",
                content_type=request.content_type or "",
                top_k=request.top_k,
            )
            return KnowledgeSearchResponse(
                message="检索完成",
                query=request.query,
                total=len(items),
                items=items,
            )
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"检索服务暂不可用: {e}")

    @app.post("/recommend", response_model=KnowledgeSearchResponse)
    async def recommend_knowledge(
            request: KnowledgeSearchRequest,
            service: QueryService = Depends(get_query_service),
    ):
        """按年龄段、问题类型和场景推荐育儿建议/话术/科普内容。"""
        try:
            items = service.search_recommendations(
                query=request.query,
                age_group=request.age_group or "",
                problem_type=request.problem_type or "",
                scene=request.scene or "",
                content_type=request.content_type or "",
                top_k=request.top_k,
            )
            return KnowledgeSearchResponse(
                message="推荐完成",
                query=request.query,
                total=len(items),
                items=items,
            )
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"推荐服务暂不可用: {e}")

    @app.post("/cases/search", response_model=KnowledgeSearchResponse)
    async def search_cases(
            request: KnowledgeSearchRequest,
            service: QueryService = Depends(get_query_service),
    ):
        """检索亲子案例片段。"""
        try:
            items = service.search_cases(
                query=request.query,
                age_group=request.age_group or "",
                problem_type=request.problem_type or "",
                scene=request.scene or "",
                top_k=request.top_k,
            )
            return KnowledgeSearchResponse(
                message="案例检索完成",
                query=request.query,
                total=len(items),
                items=items,
            )
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"案例检索服务暂不可用: {e}")

    @app.delete("/history/{session_id}")
    async def clear_chat_history(
            session_id: str,
            service: QueryService = Depends(get_query_service),
    ):
        """清除对话历史"""
        try:
            count = service.clear_history(session_id)
            return {"message": "History cleared", "deleted_count": count}
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"存储服务暂不可用: {e}")


if __name__ == '__main__':
    uvicorn.run(create_app(), host="0.0.0.0", port=8001)
