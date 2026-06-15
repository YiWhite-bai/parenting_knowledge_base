"""RRF (Reciprocal Rank Fusion) 融合节点。

职责：将多路检索结果用 RRF 算法融合，得到一个统一排序的文档列表。
"""

from typing import List, Dict, Any

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState


class RrfNode(BaseNode):
    """RRF 融合节点。"""

    name = "rrf_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        vec_chunks = state.get("embedding_chunks", [])
        hyde_chunks = state.get("hyde_embedding_chunks", [])

        rrf_k = self.config.rrf_k
        max_results = self.config.rrf_max_results

        self.log_step("Step 1", f"RRF 融合，k={rrf_k}，源1: {len(vec_chunks)}，源2: {len(hyde_chunks)}")

        # 如果只有一个来源有结果，直接使用
        if not vec_chunks and not hyde_chunks:
            state["rrf_chunks"] = []
            return state
        if not vec_chunks:
            state["rrf_chunks"] = hyde_chunks[:max_results]
            return state
        if not hyde_chunks:
            state["rrf_chunks"] = vec_chunks[:max_results]
            return state

        # RRF 融合
        scores: Dict[int, float] = {}
        chunk_map: Dict[int, Dict[str, Any]] = {}

        for source_chunks in [vec_chunks, hyde_chunks]:
            for rank, chunk in enumerate(source_chunks):
                chunk_id = chunk.get("chunk_id", id(chunk))
                rrf_score = 1.0 / (rrf_k + rank + 1)
                scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
                if chunk_id not in chunk_map:
                    chunk_map[chunk_id] = chunk

        # 按 RRF 分数降序排列
        sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
        merged = []
        for cid in sorted_ids[:max_results]:
            chunk = dict(chunk_map[cid])
            chunk["rrf_score"] = scores[cid]
            merged.append(chunk)

        self.logger.info(f"RRF 融合完成，最终 {len(merged)} 个结果")
        state["rrf_chunks"] = merged

        # 思考
        task_id = state.get("task_id", "")
        if task_id and state.get("is_stream"):
            self._push_thinking(task_id, f"融合排序完成，共 {len(merged)} 条候选资料")

        return state
