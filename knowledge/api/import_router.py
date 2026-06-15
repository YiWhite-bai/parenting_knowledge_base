"""导入服务 FastAPI 应用"""

import logging
import os

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from knowledge.core.deps import get_upload_file_service
from knowledge.core.paths import get_front_page_dir
from knowledge.schema.upload_schema import (
    UploadCheckRequest,
    UploadCheckResponse,
    UploadResponse,
    TaskStatusResponse,
)
from knowledge.service.upload_service import UpLoadService
from knowledge.processor.import_process.exceptions import FileProcessingError
from knowledge.utils.task_util import get_task_info
from knowledge.utils.sse_util import sse_generator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class DirectoryImportRequest(BaseModel):
    source_dir: str = Field(..., description="待导入的目录路径")


def create_app():
    """创建 FastAPI 实例"""
    app = FastAPI(description="育儿知识库导入服务", version="v1.0")

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
        return {"flag": "success", "service": "育儿知识库导入服务"}

    @app.post("/upload/check", response_model=UploadCheckResponse)
    def upload_check_endpoint(
            request: UploadCheckRequest,
            upload_service: UpLoadService = Depends(get_upload_file_service),
    ):
        """上传前重名检查"""
        try:
            duplicated, file_title = upload_service.check_duplicate_by_filename(request.filename)
        except FileProcessingError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ConnectionError as e:
            raise HTTPException(status_code=503, detail=f"存储服务暂不可用: {e}")
        return UploadCheckResponse(duplicated=duplicated, file_title=file_title)

    @app.post("/upload", response_model=UploadResponse)
    def upload_endpoint(
            file: UploadFile,
            background_tasks: BackgroundTasks,
            upload_service: UpLoadService = Depends(get_upload_file_service),
    ):
        """上传单个文件"""
        logger.info("收到上传请求: filename=%s", file.filename)
        try:
            task_id, import_file_path, file_dir = upload_service.process_upload_file(file)
        except FileProcessingError as e:
            logger.warning("上传文件校验/保存失败: filename=%s, error=%s", file.filename, e)
            raise HTTPException(status_code=400, detail=str(e))
        except ConnectionError as e:
            logger.error("上传依赖服务不可用: filename=%s, error=%s", file.filename, e)
            raise HTTPException(status_code=503, detail=f"存储服务暂不可用: {e}")

        logger.info("上传文件已保存到本地: task_id=%s, path=%s", task_id, import_file_path)
        background_tasks.add_task(upload_service.run_import_graph, task_id, import_file_path, file_dir)
        background_tasks.add_task(upload_service.save_origin_file_to_minio, import_file_path, file.filename)
        logger.info("导入后台任务已提交: task_id=%s", task_id)
        return UploadResponse(message=f"{file.filename}文件上传成功", task_id=task_id)

    @app.post("/import/directory", response_model=UploadResponse)
    def import_directory_endpoint(
            request: DirectoryImportRequest,
            background_tasks: BackgroundTasks,
            upload_service: UpLoadService = Depends(get_upload_file_service),
    ):
        """批量导入目录中的所有育儿文档"""
        try:
            task_id, total = upload_service.submit_directory_import(request.source_dir)
        except FileProcessingError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ConnectionError as e:
            raise HTTPException(status_code=503, detail=f"存储服务暂不可用: {e}")

        background_tasks.add_task(upload_service.run_directory_import, task_id, request.source_dir)
        return UploadResponse(
            message=f"批量导入任务已提交，共 {total} 个文件",
            task_id=task_id,
        )

    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    def get_task_status_endpoint(task_id: str):
        """查询任务状态"""
        task_info = get_task_info(task_id)
        return TaskStatusResponse(**task_info)

    @app.get("/stream/{task_id}")
    async def stream_task(task_id: str, request: Request):
        """SSE 实时进度流"""
        return StreamingResponse(
            sse_generator(task_id, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )


if __name__ == '__main__':
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8000, log_level="info")
