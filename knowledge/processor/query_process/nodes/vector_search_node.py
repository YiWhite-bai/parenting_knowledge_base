"""向量检索节点。

职责：从 Milvus 中做混合向量检索（稠密 + 稀疏）。
"""

import logging
from typing import List, Dict, Any

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.exceptions import SearchError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query

logger = logging.getLogger(__name__)

OUTPUT_FIELDS = [
    "chunk_id", "content", "title", "file_title", "age_group", "content_type",
    "problem_type", "scene", "author", "source_file", "source_path",
]


class VectorSearchNode(BaseNode):
    """向量检索节点 —— 稠密 + 稀疏混合检索。"""

    name = "vector_search_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        rewritten_query = state.get("rewritten_query", state.get("original_query", ""))
        collection = self.config.chunks_collection
        limit = self.config.embedding_search_limit

        self.log_step("Step 1", f"生成查询向量, limit={limit}")
        try:
            embed_model = AIClients.get_bge_m3_client()
            embed_result = embed_model.encode_queries([rewritten_query])
        except Exception as e:
            raise SearchError(message=f"查询向量生成失败: {e}", node_name=self.name) from e

        dense_vector = embed_result["dense"][0]
        sparse_vector = self._extract_sparse(embed_result["sparse"], 0)

        self.log_step("Step 2", "执行 Milvus 混合检索")
        try:
            milvus_client = StorageClients.get_milvus_client()
            age_group = state.get("age_group", "")
            expr = self._build_age_filter(age_group)
            if expr:
                logger.info(f"应用年龄段过滤: {age_group}")

            results = self._search(milvus_client, collection, dense_vector.tolist(), sparse_vector, limit, expr)
            if expr and not self._flatten_results(results):
                logger.info("年龄段过滤无结果，自动放宽为全库检索")
                results = self._search(milvus_client, collection, dense_vector.tolist(), sparse_vector, limit, None)
        except Exception as e:
            self.logger.error(f"Milvus 检索失败: {e}")
            return {
                "search_errors": [{
                    "source": self.name,
                    "error": str(e),
                }],
                "embedding_chunks": [],
            }

        # 展平结果
        chunks = self._flatten_results(results)
        self.logger.info(f"向量检索返回 {len(chunks)} 个结果")

        # 思考
        task_id = state.get("task_id", "")
        if task_id and state.get("is_stream"):
            self._push_thinking(task_id, f"直接检索找到 {len(chunks)} 条相关育儿知识")

        return {"embedding_chunks": chunks}

    @staticmethod
    def _build_age_filter(age_group: str) -> str | None:
        if not age_group:
            return None
        escaped_age = age_group.replace("\\", "\\\\").replace('"', '\\"')
        return f'(age_group == "{escaped_age}" or age_group == "")'

    @staticmethod
    def _search(milvus_client, collection: str, dense_vector, sparse_vector, limit: int, expr: str | None):
        search_requests = create_hybrid_search_requests(
            dense_vector, sparse_vector, limit=limit, expr=expr
        )
        return execute_hybrid_search_query(
            milvus_client, collection, search_requests,
            limit=limit,
            output_fields=OUTPUT_FIELDS,
        )

    @staticmethod
    def _extract_sparse(sparse_csr, index: int) -> Dict[int, float]:
        """从 CSR 稀疏矩阵中提取指定行的稀疏向量。"""
        start = sparse_csr.indptr[index]
        end = sparse_csr.indptr[index + 1]
        token_ids = sparse_csr.indices[start:end].tolist()
        weights = sparse_csr.data[start:end].tolist()
        return dict(zip(token_ids, weights))

    @staticmethod
    def _flatten_results(results: List) -> List[Dict[str, Any]]:
        """展平 Milvus 返回的二维结果列表。"""
        flat = []
        if not results:
            return flat
        for query_results in results:
            for hit in query_results:
                entity = dict(hit.get("entity", {}))
                entity["score"] = hit.get("distance", 0)
                entity["source"] = "vector_search"
                flat.append(entity)
        return flat
