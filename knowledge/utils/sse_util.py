import asyncio
import json
import logging
import queue
import threading
import time
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import Request


class SSEEvent:
    PROGRESS = "progress"
    DELTA = "delta"
    FINAL = "final"
    THINKING = "thinking"


_SSE_QUEUE_MAX_SIZE = 500
_SSE_QUEUE_STALE_SECONDS = 600

_task_stream: Dict[str, Dict[str, Any]] = {}
_stream_lock = threading.Lock()


def _cleanup_stale_queues():
    """清理过期的 SSE 队列，防止客户端断开后队列泄漏。"""
    now = time.time()
    stale = [
        task_id
        for task_id, entry in list(_task_stream.items())
        if now - entry.get("created_at", now) > _SSE_QUEUE_STALE_SECONDS
    ]
    for task_id in stale:
        _task_stream.pop(task_id, None)
        logging.warning("清理过期 SSE 队列: task_id=%s", task_id)


def get_sse_queue(task_id: str) -> Optional[queue.Queue]:
    """获取指定任务的队列"""
    with _stream_lock:
        entry = _task_stream.get(task_id)
        return entry["queue"] if entry else None


def create_sse_queue(task_id: str) -> queue.Queue:
    """创建并注册一个新的 SSE 队列"""
    q = queue.Queue(maxsize=_SSE_QUEUE_MAX_SIZE)
    with _stream_lock:
        _task_stream[task_id] = {"queue": q, "created_at": time.time()}
    _cleanup_stale_queues()
    return q


def remove_sse_queue(task_id: str):
    """移除指定任务的队列"""
    with _stream_lock:
        _task_stream.pop(task_id, None)


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def push_sse_event(task_id: str, event: str, data: Dict[str, Any]):
    """通过 task_id 推送事件到 SSE 队列"""
    stream_queue = get_sse_queue(task_id)
    if stream_queue is None:
        return
    try:
        stream_queue.put_nowait({"event": event, "data": data})
    except queue.Full:
        logging.warning("SSE 队列已满, task_id=%s, event=%s, 丢弃事件", task_id, event)


async def sse_generator(task_id: str, request: Request) -> AsyncGenerator:
    """流式输出结果的消费者"""
    entry = _task_stream.get(task_id)
    sse_queue = entry["queue"] if entry else None

    if sse_queue is None:
        return

    loop = asyncio.get_running_loop()

    try:
        while True:
            if await request.is_disconnected():
                return
            try:
                msg = await loop.run_in_executor(None, sse_queue.get, True, 1)
                event_type = msg.get('event')
                event_data = msg.get('data')
                yield _sse_pack(event_type, event_data)
            except queue.Empty:
                logging.debug("队列为空...请稍等")
                continue
    except (ConnectionResetError, BrokenPipeError):
        return
    except asyncio.CancelledError:
        raise
    finally:
        remove_sse_queue(task_id)
