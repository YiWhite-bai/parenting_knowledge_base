"""导入流程配置模块。"""

import os
from dataclasses import dataclass, field
from typing import Optional, Set

from dotenv import load_dotenv

from knowledge.core.paths import get_env_file_path

load_dotenv(get_env_file_path())


@dataclass
class ImportConfig:
    """导入流程配置对象。"""

    max_content_length: int = 2000
    min_content_length: int = 500
    overlap_sentences: int = 1

    image_extensions: Set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    )

    openai_api_base: str = field(default_factory=lambda: os.getenv("OPENAI_API_BASE", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    default_model: str = field(default_factory=lambda: os.getenv("LLM_DEFAULT_MODEL", ""))

    milvus_url: str = field(default_factory=lambda: os.getenv("MILVUS_URL", ""))
    chunks_collection: str = field(default_factory=lambda: os.getenv("CHUNKS_COLLECTION", ""))

    minio_endpoint: str = field(default_factory=lambda: os.getenv("MINIO_ENDPOINT", ""))
    minio_access_key: str = field(default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", ""))
    minio_secret_key: str = field(default_factory=lambda: os.getenv("MINIO_SECRET_KEY", ""))
    minio_bucket: str = field(default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", ""))
    minio_secure: bool = False

    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024")))
    embedding_batch_size: int = 8

    @classmethod
    def from_env(cls) -> "ImportConfig":
        return cls()

    def get_minio_base_url(self):
        base_protocol = "https://" if self.minio_secure else "http://"
        return base_protocol + f"{self.minio_endpoint}"


_config: Optional[ImportConfig] = None


def get_config() -> ImportConfig:
    """获取导入流程配置单例。"""
    global _config
    if _config is None:
        _config = ImportConfig.from_env()
    return _config
