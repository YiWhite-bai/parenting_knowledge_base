import threading
from typing import Optional
import logging

logger = logging.getLogger(__name__)

from minio import Minio
from pymilvus import MilvusClient
from pymongo import MongoClient
from pymongo.database import Database
from dotenv import load_dotenv
from knowledge.core.paths import get_env_file_path
from knowledge.utils.client.base import BaseClientManager

load_dotenv(get_env_file_path())


class StorageClients(BaseClientManager):
    """存储类客户端：MinIO、Milvus、MongoDB"""

    _minio_client: Optional[Minio] = None
    _minio_lock = threading.Lock()

    _milvus_client: Optional[MilvusClient] = None
    _milvus_lock = threading.Lock()

    _mongo_db: Optional[Database] = None
    _mongo_client: Optional[MongoClient] = None
    _mongo_lock = threading.Lock()

    # ── MinIO ──
    @classmethod
    def get_minio_client(cls) -> Minio:
        return cls._get_or_create("_minio_client", cls._minio_lock, cls._create_minio_client)

    @classmethod
    def _create_minio_client(cls) -> Minio:
        try:
            endpoint = cls._require_env("MINIO_ENDPOINT")
            access_key = cls._require_env("MINIO_ACCESS_KEY")
            secret_key = cls._require_env("MINIO_SECRET_KEY")
            bucket_name = cls._require_env("MINIO_BUCKET_NAME")

            secure = cls._env_bool("MINIO_SECURE", default=False)
            client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
                logger.info(f"MinIO bucket '{bucket_name}' 已自动创建")
            else:
                logger.info(f"MinIO bucket '{bucket_name}' 已存在")

            logger.info(f"MinIO 客户端初始化成功 (endpoint={endpoint})")
            return client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"MinIO 客户端创建失败: {e}")
            raise ConnectionError(f"MinIO 连接失败: {e}") from e

    # ── Milvus ──
    @classmethod
    def get_milvus_client(cls) -> MilvusClient:
        return cls._get_or_create("_milvus_client", cls._milvus_lock, cls._create_milvus_client)

    @classmethod
    def _create_milvus_client(cls) -> MilvusClient:
        try:
            milvus_uri = cls._require_env('MILVUS_URL')
            milvus_client = MilvusClient(uri=milvus_uri, timeout=30)
            return milvus_client
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"Milvus 客户端创建失败: {e}")
            raise ConnectionError(f"Milvus 连接失败: {e}") from e

    # ── MongoDB ──
    @classmethod
    def get_mongo_db(cls) -> Database:
        return cls._get_or_create("_mongo_db", cls._mongo_lock, cls._create_mongo_db)

    @classmethod
    def _create_mongo_db(cls) -> Database:
        try:
            mongo_url = cls._require_env("MONGO_URL")
            db_name = cls._require_env("MONGO_DB_NAME")

            client = MongoClient(mongo_url)
            cls._mongo_client = client
            db = client[db_name]

            logger.info(f"MongoDB 客户端初始化成功 (db={db_name})")
            return db
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"MongoDB 客户端创建失败: {e}")
            raise ConnectionError(f"MongoDB 连接失败: {e}") from e

    # ── MinIO 预签名 URL ──
    @classmethod
    def get_presigned_url(cls, full_url: str, expires_hours: int = 24) -> str:
        """将 MinIO 直接访问 URL 转换为预签名 URL。"""
        try:
            import os
            from datetime import timedelta
            from urllib.parse import urlparse

            parsed = urlparse(full_url)
            endpoint = os.getenv("MINIO_ENDPOINT", "")
            if not endpoint:
                return full_url

            base_url = f"http://{endpoint}"
            if not full_url.startswith(base_url):
                return full_url

            path_parts = parsed.path.split("/", 2)
            if len(path_parts) < 3 or not path_parts[1] or not path_parts[2]:
                return full_url

            bucket_name = path_parts[1]
            object_key = path_parts[2]

            ext = object_key.rsplit(".", 1)[-1].lower() if "." in object_key else ""
            content_type_map = {
                "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp",
                "svg": "image/svg+xml",
            }
            resp_headers = {"response-content-disposition": "inline"}
            if ext in content_type_map:
                resp_headers["response-content-type"] = content_type_map[ext]

            minio_client = cls.get_minio_client()
            presigned = minio_client.presigned_get_object(
                bucket_name=bucket_name,
                object_name=object_key,
                expires=timedelta(hours=expires_hours),
                response_headers=resp_headers,
            )
            return presigned
        except Exception as e:
            logger.warning(f"生成预签名 URL 失败 ({full_url}): {e}")
            return full_url

    # ── 生命周期 ──
    @classmethod
    def shutdown(cls) -> None:
        """关闭所有存储客户端连接。"""
        for name, client in [
            ("MinIO", cls._minio_client),
            ("Milvus", cls._milvus_client),
            ("MongoDB", cls._mongo_client),
        ]:
            if client is None:
                continue
            try:
                close_method = getattr(client, "close", None) or getattr(client, "disconnect", None)
                if close_method:
                    close_method()
                logger.info(f"{name} 客户端已关闭")
            except Exception as e:
                logger.warning(f"{name} 客户端关闭异常: {e}")
        cls._minio_client = None
        cls._milvus_client = None
        cls._mongo_db = None
        cls._mongo_client = None
