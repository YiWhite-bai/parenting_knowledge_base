from typing import Dict, List

from pydantic import BaseModel, Field


class UploadCheckRequest(BaseModel):
    """上传前重名检查请求"""
    filename: str = Field(..., description="待上传文件名")


class UploadCheckResponse(BaseModel):
    """上传前重名检查响应"""
    duplicated: bool = Field(..., description="是否已存在同标题文档")
    file_title: str = Field(..., description="提取出的文件标题")


class UploadResponse(BaseModel):
    """文件上传响应"""
    message: str = Field(..., description="响应消息")
    task_id: str = Field(..., description="任务ID")


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    status: str = Field(..., description="任务状态")
    running_list: List[str] = Field(..., description="正在运行节点列表")
    done_list: List[str] = Field(..., description="已完成节点列表")
    durations: Dict[str, float] = Field(default_factory=dict, description="各节点耗时(秒)")
    error: str = Field(default="", description="失败原因")
    total_files: int = Field(default=0, description="批量导入总文件数")
    imported_count: int = Field(default=0, description="批量导入成功文件数")
    skipped_count: int = Field(default=0, description="批量导入跳过/失败文件数")
    current_file: str = Field(default="", description="当前正在处理的文件")
    errors: List[str] = Field(default_factory=list, description="批量导入错误明细")
