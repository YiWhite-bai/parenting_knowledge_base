"""查询流程主图编排。

流程结构:
    search_router_node
         │
         ├── route_action="answer" ──> answer_output_node
         │
         └── route_action="retrieve"
              │
              ├── vector_search_node ──┐
              ├── hyde_search_node   ──┤→ join → rrf_node → reranker_node → answer_output_node
"""

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from knowledge.processor.query_process.nodes.answer_output_node import AnswerOutputNode
from knowledge.processor.query_process.nodes.hyde_search_node import HyDeSearchNode
from knowledge.processor.query_process.nodes.reranker_node import RerankerNode
from knowledge.processor.query_process.nodes.rrf_node import RrfNode
from knowledge.processor.query_process.nodes.search_router_node import SearchRouterNode
from knowledge.processor.query_process.nodes.vector_search_node import VectorSearchNode
from knowledge.processor.query_process.state import QueryGraphState

load_dotenv()


def route_after_search_router(state: QueryGraphState) -> str:
    """根据检索路由节点的决策决定下一步。"""
    return state.get("route_action", "retrieve")


def create_query_graph() -> CompiledStateGraph:
    """创建查询流程图。"""
    workflow = StateGraph(QueryGraphState)

    nodes = {
        "search_router_node": SearchRouterNode(),
        "multi_search": lambda x: x,
        "vector_search_node": VectorSearchNode(),
        "hyde_search_node": HyDeSearchNode(),
        "join": lambda x: x,
        "rrf_node": RrfNode(),
        "reranker_node": RerankerNode(),
        "answer_output_node": AnswerOutputNode(),
    }

    for name, node in nodes.items():
        workflow.add_node(name, node)

    # 入口
    workflow.set_entry_point("search_router_node")

    # 检索路由分流
    workflow.add_conditional_edges(
        "search_router_node",
        route_after_search_router,
        {"retrieve": "multi_search", "answer": "answer_output_node"},
    )

    # 并行检索分发
    workflow.add_edge("multi_search", "vector_search_node")
    workflow.add_edge("multi_search", "hyde_search_node")

    # 并行汇合
    workflow.add_edge("vector_search_node", "join")
    workflow.add_edge("hyde_search_node", "join")

    # 顺序收口
    workflow.add_edge("join", "rrf_node")
    workflow.add_edge("rrf_node", "reranker_node")
    workflow.add_edge("reranker_node", "answer_output_node")
    workflow.add_edge("answer_output_node", END)

    return workflow.compile()


query_app = create_query_graph()
