"""HyDE (假设性文档嵌入) 检索节点。

职责：让 LLM 先生成一段假设性的育儿知识文档，然后用这段文档的向量去 Milvus 检索。
"""

import logging
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.exceptions import SearchError
from knowledge.prompts.query_prompt import USER_HYDE_PROMPT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query

logger = logging.getLogger(__name__)

OUTPUT_FIELDS = [
    "chunk_id", "content", "title", "file_title", "age_group", "content_type",
    "problem_type", "scene", "author", "source_file", "source_path",
]


class HyDeSearchNode(BaseNode):
    """HyDE 检索节点 —— 假设性文档嵌入检索。"""

    name = "hyde_search_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        rewritten_query = state.get("rewritten_query", state.get("original_query", ""))
        collection = self.config.chunks_collection
        limit = self.config.hyde_search_limit

        self.log_step("Step 1", "生成 HyDE 假设性文档")
        age_group = state.get("age_group", "")
        problem_type = state.get("problem_type", "")

        age_hint = f"适用年龄段：{age_group}" if age_group else ""
        problem_hint = f"，问题类型：{problem_type}" if problem_type else ""
        hyde_prompt = USER_HYDE_PROMPT_TEMPLATE.format(
            age_hint=age_hint,
            problem_hint=problem_hint,
            rewritten_query=rewritten_query,
        )

        try:
            llm = AIClients.get_llm_client(response_format=False)
            response = llm.invoke([HumanMessage(content=hyde_prompt)])
            hyde_text = response.content
            self.logger.info(f"HyDE 文档生成成功，长度: {len(hyde_text)}")
        except Exception as e:
            self.logger.error(f"HyDE 文档生成失败: {e}")
            return {"hyde_embedding_chunks": []}

        self.log_step("Step 2", "对 HyDE 文档生成向量并检索")
        try:
            embed_model = AIClients.get_bge_m3_client()
            embed_result = embed_model.encode_queries([hyde_text])
        except Exception as e:
            raise SearchError(message=f"HyDE 查询向量生成失败: {e}", node_name=self.name) from e

        dense_vector = embed_result["dense"][0]
        sparse_vector = self._extract_sparse(embed_result["sparse"], 0)

        try:
            milvus_client = StorageClients.get_milvus_client()
            age_group = state.get("age_group", "")
            expr = self._build_age_filter(age_group)
            results = self._search(milvus_client, collection, dense_vector.tolist(), sparse_vector, limit, expr)
            if expr and not self._flatten_results(results):
                self.logger.info("HyDE 年龄段过滤无结果，自动放宽为全库检索")
                results = self._search(milvus_client, collection, dense_vector.tolist(), sparse_vector, limit, None)
        except Exception as e:
            self.logger.error(f"HyDE Milvus 检索失败: {e}")
            return {"hyde_embedding_chunks": []}

        chunks = self._flatten_results(results)
        self.logger.info(f"HyDE 检索返回 {len(chunks)} 个结果")

        # 思考
        task_id = state.get("task_id", "")
        if task_id and state.get("is_stream"):
            self._push_thinking(task_id, f"语义扩展检索找到 {len(chunks)} 条参考资料")

        return {"hyde_embedding_chunks": chunks}

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
        start = sparse_csr.indptr[index]
        end = sparse_csr.indptr[index + 1]
        token_ids = sparse_csr.indices[start:end].tolist()
        weights = sparse_csr.data[start:end].tolist()
        return dict(zip(token_ids, weights))

    @staticmethod
    def _flatten_results(results: List) -> List[Dict[str, Any]]:
        flat = []
        if not results:
            return flat
        for query_results in results:
            for hit in query_results:
                entity = dict(hit.get("entity", {}))
                entity["score"] = hit.get("distance", 0)
                entity["source"] = "hyde_search"
                flat.append(entity)
        return flat
