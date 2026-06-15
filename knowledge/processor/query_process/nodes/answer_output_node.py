"""答案生成节点。

职责：基于检索到的文档，使用 LLM 生成最终答案。支持流式和非流式输出。
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query_prompt import ANSWER_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.utils.task_util import set_task_result
from knowledge.utils.mongo_history_util import save_chat_message

logger = logging.getLogger(__name__)


class AnswerOutputNode(BaseNode):
    """答案生成节点 —— 含对话历史保存。"""

    name = "answer_output_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        session_id = state.get("session_id", "")
        task_id = state.get("task_id", "")
        is_stream = state.get("is_stream", False)
        original_query = state.get("original_query", "")
        rewritten_query = state.get("rewritten_query", original_query)
        reranked_docs = state.get("reranked_docs", [])

        # 如果 route_action 为 answer 且已有预置答案，直接输出
        if state.get("route_action") == "answer" and state.get("answer"):
            self.logger.info("短路输出已有答案")
            answer = state["answer"]
            self._emit_answer(task_id, answer, is_stream)
            self._save_to_history(state, answer)
            return state

        # 拼接上下文
        context = self._build_context(reranked_docs)
        if not context:
            answer = "暂未找到与您问题相关的育儿资料。建议换一种表述方式，或提供更多细节（如年龄段、具体场景）。"
            state["answer"] = answer
            self._emit_answer(task_id, answer, is_stream)
            self._save_to_history(state, answer)
            return state

        # 判断检索来源
        retrieval_source = "育儿知识库"
        age_group = state.get("age_group", "")
        problem_type = state.get("problem_type", "")
        query_domain = state.get("query_domain", "parenting")
        history = state.get("history", [])

        current_date = datetime.now().strftime("%Y年%m月%d日")
        history_text = self._format_history(history[-6:])

        prompt = ANSWER_PROMPT.format(
            current_date=current_date,
            context=context,
            history=history_text,
            query_domain=query_domain,
            age_group=age_group or "未指定",
            problem_type=problem_type or "未分类",
            retrieval_source=retrieval_source,
            question=original_query,
        )
        state["prompt"] = prompt

        self.log_step("Step 1", f"调用 LLM 生成答案, is_stream={is_stream}")

        # 思考
        if task_id and is_stream:
            doc_count = len(reranked_docs)
            self._push_thinking(task_id,
                f"根据 {doc_count} 条参考资料生成回答..." if doc_count else "未找到相关资料，基于通用知识回答..."
            )

        try:
            llm = AIClients.get_llm_client(response_format=False)

            if is_stream:
                # 流式输出
                full_answer = ""
                for chunk in llm.stream(prompt):
                    delta = chunk.content if hasattr(chunk, "content") else str(chunk)
                    full_answer += delta
                    push_sse_event(task_id, SSEEvent.DELTA, {"delta": delta})

                state["answer"] = full_answer
                self._emit_answer(task_id, full_answer, is_stream, emit_delta=False)
            else:
                response = llm.invoke(prompt)
                state["answer"] = response.content
                self._emit_answer(task_id, state["answer"], is_stream)
        except Exception as e:
            self.logger.error(f"LLM 答案生成失败: {e}")
            state["answer"] = f"答案生成失败: {e}"
            self._emit_answer(task_id, state["answer"], is_stream)

        # 保存到 MongoDB 历史
        self._save_to_history(state, state["answer"])
        return state

    @staticmethod
    def _emit_answer(task_id: str, answer: str, is_stream: bool, emit_delta: bool = True) -> None:
        if task_id:
            set_task_result(task_id, "answer", answer)
        if is_stream:
            if emit_delta:
                push_sse_event(task_id, SSEEvent.DELTA, {"delta": answer})
            push_sse_event(task_id, SSEEvent.FINAL, {"answer": answer})

    def _build_context(self, reranked_docs: List[Dict[str, Any]]) -> str:
        """构建 LLM 上下文。"""
        if not reranked_docs:
            return ""

        max_chars = self.config.max_context_chars
        parts = []
        total = 0
        for i, doc in enumerate(reranked_docs, 1):
            content = doc.get("content", "")
            title = doc.get("title", "")
            file_title = doc.get("file_title", "")
            age_group = doc.get("age_group", "")
            content_type = doc.get("content_type", "")
            problem_type = doc.get("problem_type", "")
            scene = doc.get("scene", "")
            author = doc.get("author", "")
            source_file = doc.get("source_file", "")
            source_path = doc.get("source_path", "")

            header_title = title or file_title or f"检索片段{i}"
            header = f"[文档{i}] {header_title}"
            meta_parts = []
            if file_title:
                meta_parts.append(f"文章标题: {file_title}")
            if author:
                meta_parts.append(f"作者: {author}")
            if age_group:
                meta_parts.append(f"年龄段: {age_group}")
            if content_type:
                meta_parts.append(f"内容类型: {content_type}")
            if problem_type:
                meta_parts.append(f"问题类型: {problem_type}")
            if scene:
                meta_parts.append(f"场景: {scene}")
            if source_file:
                meta_parts.append(f"来源文件: {source_file}")
            if source_path:
                meta_parts.append(f"来源路径: {source_path}")
            if meta_parts:
                header += f" ({', '.join(meta_parts)})"

            chunk_text = f"{header}\n{content}"
            if total + len(chunk_text) > max_chars:
                remaining = max_chars - total
                if remaining > 200:
                    parts.append(chunk_text[:remaining] + "...")
                break
            parts.append(chunk_text)
            total += len(chunk_text)

        return "\n\n---\n\n".join(parts)

    def _format_history(self, history: list) -> str:
        if not history:
            return "无历史对话"
        lines = []
        for h in history[-6:]:
            role = "家长" if h.get("role") == "user" else "助手"
            text = h.get("text", "")
            lines.append(f"{role}: {text[:200]}")
        return "\n".join(lines)

    def _save_to_history(self, state: QueryGraphState, answer: str) -> None:
        """保存到 MongoDB 对话历史。"""
        try:
            session_id = state.get("session_id", "")
            original_query = state.get("original_query", "")
            rewritten_query = state.get("rewritten_query", "")

            # 保存用户问题
            save_chat_message(
                session_id=session_id,
                role="user",
                text=original_query,
                rewritten_query=rewritten_query,
                age_group=state.get("age_group", ""),
                problem_type=state.get("problem_type", ""),
                query_domain=state.get("query_domain", "parenting"),
            )
            # 保存助手回答
            save_chat_message(
                session_id=session_id,
                role="assistant",
                text=answer,
                image_urls=state.get("image_urls", []),
                query_domain=state.get("query_domain", "parenting"),
            )
        except Exception as e:
            logger.warning(f"保存对话历史失败: {e}")
