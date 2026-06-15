"""查询流程异常定义模块。"""


class QueryProcessError(Exception):
    """查询流程异常基类。"""

    def __init__(self, message: str, node_name: str = "", cause: Exception = None):
        self.node_name = node_name
        self.cause = cause
        super().__init__(message)

    def __str__(self):
        parts = []
        if self.node_name:
            parts.append(f"[{self.node_name}]")
        parts.append(super().__str__())
        if self.cause:
            parts.append(f"(原因: {self.cause})")
        return " ".join(parts)


class StateFieldError(QueryProcessError):
    """状态字段错误异常。"""


class LLMError(QueryProcessError):
    """大语言模型调用错误异常。"""


class SearchError(QueryProcessError):
    """检索错误异常。"""


class MilvusError(QueryProcessError):
    """Milvus 存储错误异常。"""


class ConfigurationError(QueryProcessError):
    """配置错误异常。"""
