from functools import lru_cache

from knowledge.service.upload_service import UpLoadService
from knowledge.service.query_service import QueryService


@lru_cache
def get_upload_file_service():
    return UpLoadService()


@lru_cache
def get_query_service():
    return QueryService()
