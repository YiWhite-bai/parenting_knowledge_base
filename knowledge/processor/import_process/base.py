"""导入流程节点基类。"""

from abc import ABC, abstractmethod
import logging
import time
from typing import Optional, TypeVar

from knowledge.processor.import_process.config import ImportConfig, get_config
from knowledge.processor.import_process.exceptions import ImportProcessError
from knowledge.utils.task_util import (
    TASK_STATUS_FAILED,
    add_done_task,
    add_node_duration,
    add_running_task,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
    set_task_result,
    update_task_status,
)
from knowledge.utils.sse_util import SSEEvent, push_sse_event

T = TypeVar("T")


class BaseNode(ABC):
    """导入流程节点基类。"""

    name: str = "base_node"

    def __init__(self, config: Optional[ImportConfig] = None):
        self.config = config or get_config()
        self.logger = logging.getLogger(f"import.{self.name}")

    def __call__(self, state: T) -> T:
        task_id = state.get("task_id")
        start_time = time.time()
        self.logger.info(f"--- {self.name} 开始 ---")

        try:
            if task_id:
                add_running_task(task_id, self.name)
                self._push_progress(task_id)

            result = self.process(state)

            if result is None:
                result = state

            if task_id:
                add_done_task(task_id, self.name)
                add_node_duration(task_id, self.name, time.time() - start_time)
                self._push_progress(task_id)

            self.logger.info(f"--- {self.name} 完成 ---")
            return result
        except Exception as e:
            if task_id:
                add_done_task(task_id, self.name)
                update_task_status(task_id, TASK_STATUS_FAILED)
                set_task_result(task_id, "error", str(e))
                self._push_progress(task_id)

            self.logger.error(f"{self.name} 执行失败: {e}")
            raise ImportProcessError(message=str(e), node_name=self.name, cause=e)

    @abstractmethod
    def process(self, state: T) -> T:
        pass

    def log_step(self, step_name: str, message: str = ""):
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)

    @staticmethod
    def _push_progress(task_id: str) -> None:
        push_sse_event(
            task_id=task_id,
            event=SSEEvent.PROGRESS,
            data={
                "status": get_task_status(task_id),
                "done_list": get_done_task_list(task_id),
                "running_list": get_running_task_list(task_id),
            },
        )


def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
