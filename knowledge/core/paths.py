import os

# 项目根目录 knowledge/
KNOWLEDGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 仓库根目录 parenting_knowledge_base/
PROJECT_ROOT = os.path.abspath(os.path.join(KNOWLEDGE_ROOT, ".."))

# 项目环境变量文件
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")

# 本地文件存储基础目录
LOCAL_BASE_DIR = os.path.join(KNOWLEDGE_ROOT, "temp_data")

# 前端页面目录（预留）
FRONT_PAGE_DIR = os.path.join(KNOWLEDGE_ROOT, "front")


def get_local_base_dir() -> str:
    """获取本地文件存储基础目录"""
    return LOCAL_BASE_DIR


def get_project_root() -> str:
    """获取项目根目录"""
    return PROJECT_ROOT


def get_env_file_path() -> str:
    """获取项目 .env 文件路径"""
    return ENV_FILE


def get_front_page_dir() -> str:
    """获取前端静态页面目录"""
    return FRONT_PAGE_DIR
