import logging
from typing import List, Dict, Any
from datetime import datetime
from bson import ObjectId
from pymongo.collection import Collection
from pymongo import DESCENDING
from pymongo.errors import PyMongoError, ConnectionFailure

from knowledge.utils.client.storage_clients import StorageClients

logger = logging.getLogger(__name__)


def _get_collection() -> Collection:
    """获取 chat_message 集合"""
    return StorageClients.get_mongo_db()["chat_message"]


def save_chat_message(
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        age_group: str = "",
        problem_type: str = "",
        image_urls: List[str] = None,
        message_id: str = None,
        query_domain: str = "general",
) -> str:
    """MongoDB的写入操作：新增或修改"""
    ts = datetime.now().timestamp()

    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query,
        "age_group": age_group,
        "problem_type": problem_type,
        "query_domain": query_domain,
        "image_urls": image_urls or [],
        "ts": ts,
    }

    collection = _get_collection()
    if message_id:
        collection.update_one(
            {"_id": ObjectId(message_id)},
            {"$set": document},
        )
        return message_id
    else:
        result = collection.insert_one(document)
        return str(result.inserted_id)


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    try:
        cursor = (
            _get_collection()
            .find({"session_id": session_id})
            .sort("ts", DESCENDING)
            .limit(limit)
        )
        return list(cursor)
    except ConnectionFailure:
        raise
    except PyMongoError as e:
        logger.error(f"Error getting recent messages: {e}")
        return []


def clear_history(session_id: str) -> int:
    try:
        result = _get_collection().delete_many({"session_id": session_id})
        logger.info(f"Deleted {result.deleted_count} messages for session {session_id}")
        return result.deleted_count
    except ConnectionFailure:
        raise
    except PyMongoError as e:
        logger.error(f"Error clearing history for session {session_id}: {e}")
        return 0
