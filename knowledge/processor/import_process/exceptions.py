"""导入流程异常定义模块。"""


class ImportProcessError(Exception):
    """导入流程异常基类。"""

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


class StateFieldError(ImportProcessError):
    """状态字段错误异常。"""

    def __init__(
        self,
        node_name: str = "",
        field_name: str = "",
        expected_type: type = None,
        message: str = "",
        cause: Exception = None,
    ):
        self.field_name = field_name
        self.expected_type = expected_type
        if not message:
            message = f"状态字段 '{field_name}' 缺失或无效"
            if expected_type:
                message += f"，期望类型: {expected_type.__name__}"
        super().__init__(message, node_name=node_name, cause=cause)


class ConfigurationError(ImportProcessError):
    """配置错误异常。"""


class FileProcessingError(ImportProcessError):
    """文件处理错误异常。"""


class PdfConversionError(FileProcessingError):
    """PDF 转换错误异常。"""


class DocumentSplitError(ImportProcessError):
    """文档切分错误异常。"""


class EmbeddingError(ImportProcessError):
    """向量化错误异常。"""


class LLMError(ImportProcessError):
    """大语言模型调用错误异常。"""


class StorageError(ImportProcessError):
    """存储层错误异常。"""


class MilvusError(StorageError):
    """Milvus 存储错误异常。"""


class MinioError(StorageError):
    """MinIO 存储错误异常。"""


class ValidationError(ImportProcessError):
    """数据校验错误异常。"""
