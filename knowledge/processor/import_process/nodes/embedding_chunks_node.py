"""文档切片向量嵌入节点。"""

from typing import Any, Dict, List

from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from scipy.sparse import csr_matrix

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import EmbeddingError, StateFieldError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients


class EmbeddingChunksNode(BaseNode):
    """文档切片混合向量嵌入节点。"""

    name = "embedding_chunks_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("Step 1", "校验 chunks 的数据结构")
        validated_chunks = self._validate_state(state)

        if not validated_chunks:
            self.log_step("Step 1-1", "chunks 为空列表，跳过嵌入")
            state["chunks"] = []
            return state

        self.log_step("Step 2", "获取 BGE-M3 嵌入模型客户端")
        try:
            embed_model = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            raise EmbeddingError(
                message=f"BGE-M3 嵌入模型客户端创建失败: {str(e)}",
                node_name=self.name,
            ) from e

        batch_size = self.config.embedding_batch_size
        total = len(validated_chunks)
        self.log_step("Step 3", f"开始批量嵌入，总数: {total}, 批次大小: {batch_size}")

        final_chunks: List[Dict[str, Any]] = []
        for start_idx in range(0, total, batch_size):
            batch_chunks = validated_chunks[start_idx:start_idx + batch_size]
            end_idx = start_idx + len(batch_chunks)
            self.logger.info(f"嵌入批次进度 [{start_idx}-{end_idx}) / {total}")
            embedded_batch = self._embed_chunks(batch_chunks, embed_model)
            final_chunks.extend(embedded_batch)

        self.log_step("Step 4", f"向量嵌入完成，共处理 {len(final_chunks)} 个 chunks")
        state["chunks"] = final_chunks
        return state

    def _validate_state(self, state: ImportGraphState) -> List[Dict[str, Any]]:
        chunks = state.get("chunks")
        if not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)
        if len(chunks) == 0:
            return chunks

        required_fields = {"content"}
        for index, chunk in enumerate(chunks, start=1):
            if not isinstance(chunk, dict):
                raise ValidationError(
                    message=f"[chunk_{index}] 类型不匹配，期望 dict，实际为 {type(chunk).__name__}",
                    node_name=self.name,
                )
            missing = required_fields - chunk.keys()
            if missing:
                raise ValidationError(
                    message=f"[chunk_{index}] 缺少必需字段: {missing}",
                    node_name=self.name,
                )
        return chunks

    def _embed_chunks(self, batch_chunks: List[Dict[str, Any]], embed_model: BGEM3EmbeddingFunction) -> List[Dict[str, Any]]:
        # 嵌入文本：拼接 age_group + content_type + problem_type + content（育儿特有）
        embedding_documents = []
        for chunk in batch_chunks:
            age_group = chunk.get("age_group", "")
            content_type = chunk.get("content_type", "")
            problem_type = chunk.get("problem_type", "")
            scene = chunk.get("scene", "")
            content = chunk.get("content", "")
            prefix_parts = [p for p in [age_group, content_type, problem_type, scene] if p]
            prefix = f"{' '.join(prefix_parts)}\n" if prefix_parts else ""
            embedding_documents.append(f"{prefix}{content}")

        try:
            embed_result = embed_model.encode_documents(embedding_documents)
        except Exception as e:
            raise EmbeddingError(
                message=f"BGE-M3 嵌入调用失败: {str(e)}",
                node_name=self.name,
            ) from e

        if not embed_result:
            raise EmbeddingError(message="BGE-M3 嵌入返回结果为空", node_name=self.name)

        dense_vectors = embed_result.get("dense")
        sparse_csr = embed_result.get("sparse")
        if dense_vectors is None:
            raise EmbeddingError(message="BGE-M3 嵌入结果缺少稠密向量(dense)", node_name=self.name)
        if sparse_csr is None:
            raise EmbeddingError(message="BGE-M3 嵌入结果缺少稀疏向量(sparse)", node_name=self.name)

        actual_batch = len(dense_vectors)
        expected_batch = len(batch_chunks)
        if actual_batch != expected_batch:
            raise EmbeddingError(
                message=f"BGE-M3 返回的向量数量({actual_batch})与输入文档数量({expected_batch})不一致",
                node_name=self.name,
            )

        result_chunks = []
        for i, chunk in enumerate(batch_chunks):
            new_chunk = dict(chunk)
            new_chunk["dense_vector"] = dense_vectors[i].tolist()
            new_chunk["sparse_vector"] = self._extract_sparse_vector(sparse_csr, i)
            result_chunks.append(new_chunk)

        return result_chunks

    @staticmethod
    def _extract_sparse_vector(sparse_csr: csr_matrix, index: int) -> Dict[int, float]:
        start_index = sparse_csr.indptr[index]
        end_index = sparse_csr.indptr[index + 1]
        token_ids = sparse_csr.indices[start_index:end_index].tolist()
        weights = sparse_csr.data[start_index:end_index].tolist()
        return dict(zip(token_ids, weights))
