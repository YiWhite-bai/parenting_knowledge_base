"""查询业务服务 - 育儿知识库版本"""

import uuid
import logging
from typing import Dict, Any, List, Optional

from knowledge.processor.query_process.config import get_config as get_query_config
from knowledge.processor.query_process.main_graph import query_app
from knowledge.utils.task_util import (
    update_task_status, get_task_result as _get_task_result, get_task_status,
    get_done_task_list, get_running_task_list, get_task_info, set_task_result,
    TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED,
)
from knowledge.utils.sse_util import create_sse_queue, push_sse_event
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.mongo_history_util import get_recent_messages, clear_history as _clear_history
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query

logger = logging.getLogger(__name__)

_SEARCH_OUTPUT_FIELDS = [
    "chunk_id", "content", "title", "file_title", "age_group", "content_type",
    "problem_type", "scene", "author", "source_file", "source_path",
]

_RECOMMEND_CONTENT_TYPES = ["育儿建议", "专家文章", "沟通话术", "知识科普"]
_CASE_CONTENT_TYPES = ["亲子案例"]


class QueryService:

    def generate_session_id(self) -> str:
        return str(uuid.uuid4())

    def generate_task_id(self) -> str:
        return str(uuid.uuid4())

    def submit_query(self, task_id: str, is_stream: bool):
        """提交查询任务"""
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        if is_stream:
            create_sse_queue(task_id)

    def run_query_graph(self, session_id: str, task_id: str, query: str, is_stream: bool):
        """执行 LangGraph 查询流程。"""
        update_task_status(task_id, TASK_STATUS_PROCESSING)

        # 加载对话历史
        try:
            records = get_recent_messages(session_id, limit=10)
            history = [
                {
                    "role": r.get("role", ""),
                    "text": r.get("text", ""),
                    "rewritten_query": r.get("rewritten_query", ""),
                }
                for r in reversed(records)
            ]
        except Exception as e:
            logger.warning(f"加载对话历史失败: {e}")
            history = []

        try:
            default_state = {
                "original_query": query,
                "session_id": session_id,
                "task_id": task_id,
                "is_stream": is_stream,
                "history": history,
                "rewritten_query": query,
            }
            result_state = query_app.invoke(default_state)

            if isinstance(result_state, dict):
                answer = result_state.get("answer", "")
                if answer and not _get_task_result(task_id, "answer", ""):
                    set_task_result(task_id, "answer", answer)
                image_urls = result_state.get("image_urls", [])
                if image_urls and not _get_task_result(task_id, "image_urls", []):
                    set_task_result(task_id, "image_urls", image_urls)
                reranked_docs = result_state.get("reranked_docs", [])
                if reranked_docs and not _get_task_result(task_id, "reranked_docs", []):
                    set_task_result(task_id, "reranked_docs", self._sanitize_docs(reranked_docs))

            update_task_status(task_id, TASK_STATUS_COMPLETED)
            if is_stream:
                push_sse_event(task_id, "progress", {
                    "status": get_task_status(task_id),
                    "done_list": get_done_task_list(task_id),
                    "running_list": get_running_task_list(task_id),
                })
        except Exception as e:
            logger.error(f"查询流程执行失败: {e}", exc_info=True)
            set_task_result(task_id, "error", str(e))
            update_task_status(task_id, TASK_STATUS_FAILED)
            if is_stream:
                push_sse_event(task_id, "final", {"error": str(e)})
        finally:
            if is_stream:
                push_sse_event(task_id, "progress", {
                    "status": get_task_status(task_id),
                    "done_list": get_done_task_list(task_id),
                    "running_list": get_running_task_list(task_id),
                })

    def get_task_payload(self, task_id: str) -> Dict[str, Any]:
        task_info = get_task_info(task_id)
        task_info["answer"] = _get_task_result(task_id, "answer", "")
        task_info["error"] = _get_task_result(task_id, "error", "")
        task_info["image_urls"] = self._to_presigned_urls(
            _get_task_result(task_id, "image_urls", [])
        )
        task_info["reranked_docs"] = _get_task_result(task_id, "reranked_docs", [])
        return task_info

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        records = get_recent_messages(session_id, limit=limit)
        return [
            {
                "_id": str(r.get("_id", "")),
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "age_group": r.get("age_group", ""),
                "problem_type": r.get("problem_type", ""),
                "query_domain": r.get("query_domain", ""),
                "image_urls": self._to_presigned_urls(r.get("image_urls", [])),
                "ts": r.get("ts"),
            }
            for r in records
        ]

    def clear_history(self, session_id: str) -> int:
        return _clear_history(session_id)

    def search_recommendations(
            self,
            query: str,
            *,
            age_group: str = "",
            problem_type: str = "",
            scene: str = "",
            content_type: str = "",
            top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """按育儿元数据检索推荐内容。"""
        content_types = [content_type] if content_type else _RECOMMEND_CONTENT_TYPES
        return self._metadata_search(
            query=query,
            age_group=age_group,
            problem_type=problem_type,
            scene=scene,
            content_types=content_types,
            top_k=top_k,
            allow_unfiltered_fallback=True,
        )

    def search_cases(
            self,
            query: str,
            *,
            age_group: str = "",
            problem_type: str = "",
            scene: str = "",
            top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """检索亲子案例内容。"""
        return self._metadata_search(
            query=query,
            age_group=age_group,
            problem_type=problem_type,
            scene=scene,
            content_types=_CASE_CONTENT_TYPES,
            top_k=top_k,
            allow_unfiltered_fallback=False,
        )

    def search_knowledge(
            self,
            query: str,
            *,
            age_group: str = "",
            problem_type: str = "",
            scene: str = "",
            content_type: str = "",
            top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """通用育儿知识检索。"""
        content_types = [content_type] if content_type else []
        return self._metadata_search(
            query=query,
            age_group=age_group,
            problem_type=problem_type,
            scene=scene,
            content_types=content_types,
            top_k=top_k,
            allow_unfiltered_fallback=True,
        )

    @staticmethod
    def _to_presigned_urls(urls: List[str]) -> List[str]:
        if not urls:
            return []
        try:
            return [StorageClients.get_presigned_url(u) for u in urls if u]
        except Exception as e:
            logger.warning(f"预签名 URL 转换失败: {e}")
            return [u for u in urls if u]

    def _metadata_search(
            self,
            *,
            query: str,
            age_group: str,
            problem_type: str,
            scene: str,
            content_types: List[str],
            top_k: int,
            allow_unfiltered_fallback: bool,
    ) -> List[Dict[str, Any]]:
        search_text = self._build_metadata_search_text(query, age_group, problem_type, scene, content_types)
        dense_vector, sparse_vector = self._embed_query(search_text)
        milvus_client = StorageClients.get_milvus_client()
        collection = get_query_config().chunks_collection

        expr_candidates = self._build_expr_candidates(
            age_group=age_group,
            problem_type=problem_type,
            scene=scene,
            content_types=content_types,
            allow_unfiltered_fallback=allow_unfiltered_fallback,
        )

        last_error: Optional[Exception] = None
        for expr in expr_candidates:
            try:
                search_requests = create_hybrid_search_requests(
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    expr=expr,
                    limit=top_k,
                )
                raw_results = execute_hybrid_search_query(
                    milvus_client=milvus_client,
                    collection_name=collection,
                    search_requests=search_requests,
                    limit=top_k,
                    output_fields=_SEARCH_OUTPUT_FIELDS,
                )
                docs = self._flatten_search_results(raw_results, top_k)
                if docs:
                    return docs
            except Exception as e:
                last_error = e
                logger.warning(f"元数据检索失败，尝试放宽过滤条件: expr={expr}, error={e}")

        if last_error:
            logger.warning(f"元数据检索最终无结果，最后错误: {last_error}")
        return []

    @staticmethod
    def _build_metadata_search_text(
            query: str,
            age_group: str,
            problem_type: str,
            scene: str,
            content_types: List[str],
    ) -> str:
        hints = [query]
        if age_group:
            hints.append(f"年龄段：{age_group}")
        if problem_type:
            hints.append(f"问题类型：{problem_type}")
        if scene:
            hints.append(f"场景：{scene}")
        if content_types:
            hints.append(f"内容类型：{' '.join(content_types)}")
        return "\n".join(hints)

    @staticmethod
    def _embed_query(query_text: str):
        embed_model = AIClients.get_bge_m3_client()
        embed_result = embed_model.encode_queries([query_text])
        dense_vector = embed_result["dense"][0].tolist()
        sparse_vector = QueryService._extract_sparse(embed_result["sparse"], 0)
        return dense_vector, sparse_vector

    @staticmethod
    def _extract_sparse(sparse_csr, index: int) -> Dict[int, float]:
        start = sparse_csr.indptr[index]
        end = sparse_csr.indptr[index + 1]
        token_ids = sparse_csr.indices[start:end].tolist()
        weights = sparse_csr.data[start:end].tolist()
        return dict(zip(token_ids, weights))

    @classmethod
    def _build_expr_candidates(
            cls,
            *,
            age_group: str,
            problem_type: str,
            scene: str,
            content_types: List[str],
            allow_unfiltered_fallback: bool,
    ) -> List[Optional[str]]:
        exprs: List[Optional[str]] = []
        strict = cls._join_expr_parts([
            cls._content_type_expr(content_types),
            cls._age_group_expr(age_group),
            cls._eq_expr("problem_type", problem_type),
            cls._like_expr("scene", scene),
        ])
        no_scene = cls._join_expr_parts([
            cls._content_type_expr(content_types),
            cls._age_group_expr(age_group),
            cls._eq_expr("problem_type", problem_type),
        ])
        content_age = cls._join_expr_parts([
            cls._content_type_expr(content_types),
            cls._age_group_expr(age_group),
        ])
        content_only = cls._join_expr_parts([cls._content_type_expr(content_types)])

        for expr in [strict, no_scene, content_age, content_only]:
            if expr and expr not in exprs:
                exprs.append(expr)
        if allow_unfiltered_fallback:
            exprs.append(None)
        return exprs or [None]

    @staticmethod
    def _join_expr_parts(parts: List[Optional[str]]) -> Optional[str]:
        valid_parts = [p for p in parts if p]
        if not valid_parts:
            return None
        return " and ".join(valid_parts)

    @classmethod
    def _content_type_expr(cls, content_types: List[str]) -> Optional[str]:
        clean_types = [t.strip() for t in content_types if t and t.strip()]
        if not clean_types:
            return None
        values = ", ".join(f'"{cls._escape_expr_value(t)}"' for t in clean_types)
        return f"content_type in [{values}]"

    @classmethod
    def _age_group_expr(cls, age_group: str) -> Optional[str]:
        if not age_group:
            return None
        escaped = cls._escape_expr_value(age_group)
        return f'(age_group == "{escaped}" or age_group == "")'

    @classmethod
    def _eq_expr(cls, field: str, value: str) -> Optional[str]:
        if not value:
            return None
        escaped = cls._escape_expr_value(value)
        return f'{field} == "{escaped}"'

    @classmethod
    def _like_expr(cls, field: str, value: str) -> Optional[str]:
        if not value:
            return None
        escaped = cls._escape_expr_value(value)
        return f'{field} like "%{escaped}%"'

    @staticmethod
    def _escape_expr_value(value: str) -> str:
        return str(value).replace("\\", "\\\\").replace('"', '\\"')

    @classmethod
    def _flatten_search_results(cls, raw_results: List, top_k: int) -> List[Dict[str, Any]]:
        if not raw_results:
            return []

        docs: List[Dict[str, Any]] = []
        seen = set()
        for query_results in raw_results:
            for hit in query_results:
                entity = dict(hit.get("entity", {}))
                chunk_id = entity.get("chunk_id")
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                entity["score"] = cls._json_safe(hit.get("distance", 0.0))
                docs.append(cls._normalize_search_doc(entity))
                if len(docs) >= top_k:
                    return docs
        return docs

    @classmethod
    def _normalize_search_doc(cls, doc: Dict[str, Any]) -> Dict[str, Any]:
        normalized = cls._sanitize_docs([doc])
        if not normalized:
            return {}
        return normalized[0]

    @staticmethod
    def _sanitize_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """清理 API 返回证据，避免泄露大向量或内部中间字段。"""
        allowed_fields = {
            "chunk_id", "content", "title", "file_title", "age_group", "content_type",
            "problem_type", "scene", "author", "source_file", "source_path", "source",
            "score", "rrf_score", "rerank_score",
        }
        return [
            {key: QueryService._json_safe(value) for key, value in doc.items() if key in allowed_fields}
            for doc in docs
            if isinstance(doc, dict)
        ]

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return str(value)
        return value
