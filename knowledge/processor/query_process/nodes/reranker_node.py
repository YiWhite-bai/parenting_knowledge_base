"""Reranker 精排节点。

职责：使用 BGE-Reranker 对融合后的文档进行精细排序。
"""

from typing import List, Dict, Any

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients


class RerankerNode(BaseNode):
    """Reranker 精排节点。"""

    name = "reranker_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        chunks = state.get("rrf_chunks", [])
        if not chunks:
            self.logger.info("没有待排序的文档，跳过精排")
            state["reranked_docs"] = []
            return state

        query = state.get("rewritten_query", state.get("original_query", ""))
        max_top_k = self.config.rerank_max_top_k
        min_top_k = self.config.rerank_min_top_k
        gap_ratio = self.config.rerank_gap_ratio
        gap_abs = self.config.rerank_gap_abs

        self.log_step("Step 1", f"精排开始，输入 {len(chunks)} 个文档")

        try:
            reranker = AIClients.get_bge_m3_rerank_client()
        except ConnectionError as e:
            self.logger.warning(f"Reranker 不可用，跳过精排: {e}")
            # Fallback: 保持原始 RRF 顺序
            for i, c in enumerate(chunks):
                c["rerank_score"] = c.get("rrf_score", 0)
            state["reranked_docs"] = chunks[:max_top_k]
            return state

        # 构造 pairs
        pairs = []
        for chunk in chunks:
            content = chunk.get("content", "")[:500]  # 取前 500 字符做精排
            pairs.append([query, content])

        try:
            scores = reranker.compute_score(pairs)
            # compute_score 可能返回单个值或列表
            if not isinstance(scores, list):
                scores = [scores]
        except Exception as e:
            self.logger.error(f"Reranker 计算失败: {e}")
            for c in chunks:
                c["rerank_score"] = c.get("rrf_score", 0)
            state["reranked_docs"] = chunks[:max_top_k]
            return state

        # 给每个 chunk 赋分
        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = scores[i] if i < len(scores) else 0.0

        # 按 rerank 分数降序，然后按 gap 策略截断
        sorted_chunks = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        selected = self._cut_by_gap(sorted_chunks, max_top_k, min_top_k, gap_ratio, gap_abs)

        self.logger.info(f"精排完成，选择 {len(selected)} 个文档")
        state["reranked_docs"] = selected

        # 思考
        task_id = state.get("task_id", "")
        if task_id and state.get("is_stream"):
            self._push_thinking(task_id, f"精排选取最相关的 {len(selected)} 条育儿知识")

        return state

    @staticmethod
    def _cut_by_gap(chunks: List[Dict], max_k: int, min_k: int,
                    gap_ratio: float, gap_abs: float) -> List[Dict]:
        """按分数 gap 截断：相邻文档分数差距过大时截断。"""
        if len(chunks) <= min_k:
            return chunks
        selected = chunks[:min_k]
        for i in range(min_k, min(len(chunks), max_k)):
            prev_score = chunks[i - 1]["rerank_score"]
            curr_score = chunks[i]["rerank_score"]
            if prev_score - curr_score > max(gap_abs, prev_score * gap_ratio):
                break
            selected.append(chunks[i])
        return selected
