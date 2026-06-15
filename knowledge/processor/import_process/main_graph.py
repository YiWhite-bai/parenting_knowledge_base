"""导入流程主图编排模块。

去掉 md_to_img_node 和 knowledge_graph_node，新增 parenting_metadata_node。
导入流程: entry → pdf_to_md → parenting_metadata → document_split → embedding_chunks → import_milvus → END
"""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from knowledge.processor.import_process.nodes.document_split_node import DocumentSplitNode
from knowledge.processor.import_process.nodes.embedding_chunks_node import EmbeddingChunksNode
from knowledge.processor.import_process.nodes.entry_node import EntryNode
from knowledge.processor.import_process.nodes.import_milvus_node import ImportMilvusNode
from knowledge.processor.import_process.nodes.parenting_metadata_node import ParentingMetadataNode
from knowledge.processor.import_process.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_process.state import ImportGraphState, create_default_state


def import_router(state: ImportGraphState) -> str:
    """根据入口节点写入的标志选择后续分支。"""
    if state.get("is_pdf_read_enabled"):
        return "pdf_to_md_node"
    if state.get("is_md_read_enabled"):
        return "parenting_metadata_node"
    return END


def build_import_graph() -> CompiledStateGraph:
    """构建并编译完整的导入流程状态图。"""
    workflow = StateGraph(ImportGraphState)

    nodes = {
        "entry_node": EntryNode(),
        "pdf_to_md_node": PdfToMdNode(),
        "parenting_metadata_node": ParentingMetadataNode(),
        "document_split_node": DocumentSplitNode(),
        "embedding_chunks_node": EmbeddingChunksNode(),
        "import_milvus_node": ImportMilvusNode(),
    }
    for name, node in nodes.items():
        workflow.add_node(name, node)

    workflow.set_entry_point("entry_node")
    workflow.add_conditional_edges(
        "entry_node",
        import_router,
        {
            "pdf_to_md_node": "pdf_to_md_node",
            "parenting_metadata_node": "parenting_metadata_node",
            END: END,
        },
    )

    # 串行连接
    workflow.add_edge("pdf_to_md_node", "parenting_metadata_node")
    workflow.add_edge("parenting_metadata_node", "document_split_node")
    workflow.add_edge("document_split_node", "embedding_chunks_node")
    workflow.add_edge("embedding_chunks_node", "import_milvus_node")
    workflow.add_edge("import_milvus_node", END)

    return workflow.compile()


import_app = build_import_graph()
