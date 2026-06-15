"""文档切片 Milvus 入库节点。

育儿版本：Schema 包含育儿元数据字段（age_group, content_type, problem_type, scene）。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pymilvus import DataType, MilvusClient

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import MilvusError, StateFieldError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.storage_clients import StorageClients


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Schema 与索引定义
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class _SCALAR_FIELD_SPC:
    field_name: str
    datatype: DataType
    max_length: Optional[int] = None


# 育儿知识库文档切片集合的固定标量字段定义
_SCALAR_FIELDS: Tuple[_SCALAR_FIELD_SPC, ...] = (
    _SCALAR_FIELD_SPC(field_name="content", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535),
    # 育儿领域新增字段
    _SCALAR_FIELD_SPC(field_name="age_group", datatype=DataType.VARCHAR, max_length=20),
    _SCALAR_FIELD_SPC(field_name="content_type", datatype=DataType.VARCHAR, max_length=20),
    _SCALAR_FIELD_SPC(field_name="problem_type", datatype=DataType.VARCHAR, max_length=50),
    _SCALAR_FIELD_SPC(field_name="scene", datatype=DataType.VARCHAR, max_length=200),
    _SCALAR_FIELD_SPC(field_name="author", datatype=DataType.VARCHAR, max_length=100),
    _SCALAR_FIELD_SPC(field_name="source_file", datatype=DataType.VARCHAR, max_length=255),
    _SCALAR_FIELD_SPC(field_name="source_path", datatype=DataType.VARCHAR, max_length=1024),
)


class _MilvusSchemaBuilder:
    @staticmethod
    def build_schema(milvus_client: MilvusClient, dim: int):
        schema = milvus_client.create_schema(enable_dynamic_field=True)

        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        for spec in _SCALAR_FIELDS:
            kwargs: Dict[str, Any] = {"field_name": spec.field_name, "datatype": spec.datatype}
            if spec.max_length:
                kwargs["max_length"] = spec.max_length
            schema.add_field(**kwargs)

        return schema


class _MilvusInserter:
    def __init__(self, milvus_client: MilvusClient, collection_name: str):
        self._milvus_client = milvus_client
        self._collection_name = collection_name

    def insert_rows(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        inserted_result = self._milvus_client.insert(collection_name=self._collection_name, data=data)
        chunk_ids = inserted_result.get("ids")
        result = []
        for chunk_id, chunk in zip(chunk_ids, data):
            new_chunk = dict(chunk)
            new_chunk["chunk_id"] = chunk_id
            result.append(new_chunk)
        return result


class _MilvusIndexBuilder:
    @staticmethod
    def build_index_params(milvus_client: MilvusClient):
        index = milvus_client.prepare_index_params()
        index.add_index(field_name="dense_vector", index_name="dense_vector_index",
                        index_type="AUTOINDEX", metric_type="COSINE")
        index.add_index(field_name="sparse_vector", index_name="sparse_vector_index",
                        index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
        return index


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主节点
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ImportMilvusNode(BaseNode):
    """文档切片 Milvus 入库主节点。"""

    name = "import_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("Step 1", "校验 chunks 数据结构和字段完整性")
        validated_chunks, dim = self._validate_state(state)

        if not validated_chunks:
            self.log_step("Step 1", "chunks 为空，跳过 Milvus 入库")
            return state

        self.log_step("Step 2", "获取 Milvus 客户端")
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            raise MilvusError(message=f"Milvus 客户端创建失败: {str(e)}", node_name=self.name) from e

        chunks_collection = self.config.chunks_collection

        self.log_step("Step 3", f"创建/确认 Milvus 集合: {chunks_collection}")
        self._create_chunks_collection(chunks_collection, milvus_client, dim)

        file_title = state.get("file_title", "")
        if file_title:
            self.log_step("Step 4", f"查重删除旧记录，file_title: {file_title}")
            self._delete_by_file_title(milvus_client, chunks_collection, file_title)

        self.log_step("Step 5", f"开始批量插入，共 {len(validated_chunks)} 条记录")
        _inserter = _MilvusInserter(milvus_client, chunks_collection)
        inserted_chunks = _inserter.insert_rows(validated_chunks)

        self.log_step("Step 6", f"Milvus 入库完成，共 {len(inserted_chunks)} 条记录")
        state["chunks"] = inserted_chunks
        return state

    def _validate_state(self, state: ImportGraphState) -> Tuple[List[Dict[str, Any]], int]:
        chunks = state.get("chunks")
        if not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)
        if not chunks:
            self.logger.info("chunks 为空列表，跳过 Milvus 入库")
            return [], 0

        validated_chunks = []
        required_scalar = {"content", "title", "parent_title", "file_title"}
        optional_scalar_defaults = {
            "age_group": "",
            "content_type": "",
            "problem_type": "",
            "scene": "",
            "author": "",
            "source_file": "",
            "source_path": "",
        }

        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValidationError(
                    message=f"chunks[{i}] 类型无效：期望 dict，实际为 {type(chunk).__name__}",
                    node_name=self.name,
                )
            missing_scalar = required_scalar - chunk.keys()
            if missing_scalar:
                self.logger.warning(f"chunks[{i}] 缺少标量字段 {missing_scalar}，已跳过")
                continue
            for field, default_value in optional_scalar_defaults.items():
                chunk.setdefault(field, default_value)
            if not chunk.get("dense_vector") or not chunk.get("sparse_vector"):
                self.logger.warning(f"chunks[{i}] 缺少混合向量，已跳过")
                continue
            validated_chunks.append(chunk)

        if not validated_chunks:
            raise ValidationError("所有 chunk 均无有效向量，无法入库", self.name)

        dim = len(validated_chunks[0]["dense_vector"])
        self.logger.info(f"有效 chunks：{len(validated_chunks)}，向量维度：{dim}")
        return validated_chunks, dim

    def _create_chunks_collection(self, chunks_collection: str, milvus_client: MilvusClient, dim: int):
        if milvus_client.has_collection(chunks_collection):
            self.logger.info(f"集合 {chunks_collection} 已存在，跳过创建")
            return
        schema = _MilvusSchemaBuilder.build_schema(milvus_client, dim)
        index_params = _MilvusIndexBuilder.build_index_params(milvus_client)
        milvus_client.create_collection(collection_name=chunks_collection, schema=schema, index_params=index_params)
        self.logger.info(f"集合创建成功: {chunks_collection}")

    def _delete_by_file_title(self, milvus_client: MilvusClient, collection_name: str, file_title: str) -> None:
        try:
            escaped = file_title.replace("\\", "\\\\").replace('"', '\\"')
            existing = milvus_client.query(
                collection_name=collection_name,
                filter=f'file_title == "{escaped}"',
                output_fields=["chunk_id"],
            )
            if existing:
                pks = [e["chunk_id"] for e in existing]
                milvus_client.delete(collection_name=collection_name, pks=pks)
                self.logger.info(f"删除旧记录 {len(pks)} 条，file_title: {file_title}")
        except Exception as e:
            self.logger.warning(f"查重删除失败: {e}")
