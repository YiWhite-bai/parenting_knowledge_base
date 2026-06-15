import logging

logger = logging.getLogger(__name__)

from typing import List, Tuple, Any, Dict, Optional, Union
from pymilvus import MilvusClient, WeightedRanker, AnnSearchRequest


def create_hybrid_search_requests(dense_vector,
                                  sparse_vector,
                                  dense_params=None,
                                  sparse_params=None,
                                  expr=None,
                                  expr_params=None,
                                  limit=5) -> List[AnnSearchRequest]:
    """
    创建混合搜索请求

    :param dense_vector: 稠密向量
    :param sparse_vector: 稀疏向量
    :param dense_params: 稠密向量搜索参数，默认为None
    :param sparse_params: 稀疏向量搜索参数，默认为None
    :param expr: 查询表达式，默认为None
    :param expr_params: 查询表达式的变量内容，默认为None
    :param limit: 返回结果数量限制，默认为5
    :return: 包含稠密和稀疏搜索请求的列表
    """
    if dense_vector is None or sparse_vector is None:
        raise ValueError("dense_vector 和 sparse_vector 不能为 None")

    try:
        if dense_params is None:
            dense_params = {"metric_type": "COSINE"}
        if sparse_params is None:
            sparse_params = {"metric_type": "IP"}

        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_vector",
            param=dense_params,
            expr=expr,
            expr_params=expr_params,
            limit=limit
        )

        sparse_req = AnnSearchRequest(
            data=[sparse_vector],
            anns_field="sparse_vector",
            param=sparse_params,
            expr=expr,
            expr_params=expr_params,
            limit=limit
        )

        return [dense_req, sparse_req]
    except Exception as e:
        raise RuntimeError(f"创建混合搜索请求失败: {e}") from e


def execute_hybrid_search_query(milvus_client: MilvusClient,
                                collection_name,
                                search_requests,
                                ranker_weights=(0.5, 0.5),
                                norm_score=True,
                                limit=5,
                                output_fields=None,
                                search_params=None):
    """执行混合搜索"""
    if milvus_client is None:
        raise ValueError("milvus_client 不能为 None")
    if search_requests is None or len(search_requests) == 0:
        raise ValueError("search_requests 不能为 None 或空列表")

    try:
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)

        if output_fields is None:
            output_fields = ["item_name"]

        res = milvus_client.hybrid_search(
            collection_name=collection_name,
            reqs=search_requests,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )

        total_hits = sum(len(hits) for hits in res) if res else 0
        logger.info(f"Milvus 混合搜索完成，共处理 {len(res) if res else 0} 个查询，总计找到 {total_hits} 个结果")
        return res
    except Exception as e:
        raise RuntimeError(f"执行Milvus混合搜索失败 (collection={collection_name}): {e}") from e


def fetch_chunks_by_chunk_ids(
        milvus_client: MilvusClient,
        collection_name: str,
        chunk_ids: List[Union[str, int]],
        *,
        output_fields: Optional[List[str]] = None,
        batch_size: int = 100,
) -> List[Dict[str, Any]]:
    """通过 chunk_id（主键）批量查询切片字段。"""
    if milvus_client is None or not collection_name or not chunk_ids:
        return []

    if output_fields is None:
        output_fields = [
            "chunk_id", "content", "title", "file_title", "age_group", "content_type",
            "problem_type", "scene", "author", "source_file", "source_path",
        ]

    results = []
    for i in range(0, len(chunk_ids), batch_size):
        batch = chunk_ids[i: i + batch_size]
        try:
            got = milvus_client.get(
                collection_name=collection_name,
                ids=batch,
                output_fields=output_fields,
            )
            if got:
                results.extend(got)
        except Exception as e:
            logger.error(f"Milvus get() 查询失败: {e}")

    return results
