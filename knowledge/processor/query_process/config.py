"""查询流程配置管理模块"""

from dataclasses import dataclass, field
from typing import Optional
import os

from dotenv import load_dotenv

from knowledge.core.paths import get_env_file_path

load_dotenv(get_env_file_path())


@dataclass
class QueryConfig:
    """查询流程配置。"""

    max_context_chars: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
    )

    rerank_max_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MAX_TOP_K", "10"))
    )
    rerank_min_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MIN_TOP_K", "3"))
    )
    rerank_gap_ratio: float = field(
        default_factory=lambda: float(os.getenv("RERANK_GAP_RATIO", "0.25"))
    )
    rerank_gap_abs: float = field(
        default_factory=lambda: float(os.getenv("RERANK_GAP_ABS", "0.5"))
    )
    rerank_local_boost: float = field(
        default_factory=lambda: float(os.getenv("RERANK_LOCAL_BOOST", "0.05"))
    )

    rrf_k: int = field(default_factory=lambda: int(os.getenv("RRF_K", "60")))
    rrf_max_results: int = field(default_factory=lambda: int(os.getenv("RRF_MAX_RESULTS", "10")))

    embedding_search_limit: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_SEARCH_LIMIT", "5"))
    )
    hyde_search_limit: int = field(
        default_factory=lambda: int(os.getenv("HYDE_SEARCH_LIMIT", "5"))
    )

    openai_api_base: str = field(default_factory=lambda: os.getenv("OPENAI_API_BASE", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    default_model: str = field(default_factory=lambda: os.getenv("LLM_DEFAULT_MODEL", ""))

    milvus_url: str = field(default_factory=lambda: os.getenv("MILVUS_URL", ""))
    chunks_collection: str = field(default_factory=lambda: os.getenv("CHUNKS_COLLECTION", ""))

    @classmethod
    def from_env(cls) -> "QueryConfig":
        return cls()


_config: Optional[QueryConfig] = None


def get_config() -> QueryConfig:
    """获取配置单例。"""
    global _config
    if _config is None:
        _config = QueryConfig.from_env()
    return _config
