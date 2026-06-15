"""检索路由节点。

职责：分析用户问题，提取年龄段、问题类型，改写查询，决定路由。
"""

import json
import logging
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query_prompt import SEARCH_ROUTER_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients

logger = logging.getLogger(__name__)


class SearchRouterNode(BaseNode):
    """问题分析 & 检索路由节点。"""

    name = "search_router_node"

    # 年龄段关键词 fallback
    AGE_PATTERNS = {
        "0-3岁": ["0-3", "1岁", "2岁", "3岁", "宝宝", "婴儿", "幼儿", "依恋", "如厕", "分离焦虑"],
        "3-6岁": ["3-6", "4岁", "5岁", "6岁", "幼儿园", "入园", "学前", "挑食", "发脾气"],
        "6-12岁": ["6-12", "7岁", "8岁", "9岁", "10岁", "小学", "作业", "专注力", "同伴"],
        "12+岁": ["12", "青春期", "顶嘴", "手机", "独立", "社交媒体"],
    }

    def process(self, state: QueryGraphState) -> QueryGraphState:
        query = state.get("original_query", "")
        history = state.get("history", [])

        self.log_step("Step 1", "使用 LLM 分析问题意图")
        try:
            llm = AIClients.get_llm_client(response_format=True)
            history_text = self._format_history(history[-6:])
            current_date = datetime.now().strftime("%Y年%m月%d日")
            user_prompt = SEARCH_ROUTER_TEMPLATE.format(
                current_date=current_date,
                history_text=history_text,
                query=query,
            )
            response = llm.invoke([HumanMessage(content=user_prompt)])
            result = json.loads(response.content)
        except Exception as e:
            logger.warning(f"LLM 问题分析失败，使用 fallback: {e}")
            result = self._fallback_analyze(query)

        state["rewritten_query"] = result.get("rewritten_query", query)
        state["age_group"] = result.get("age_group", "")
        state["problem_type"] = result.get("problem_type", "")
        state["query_domain"] = result.get("query_domain", "parenting")
        state["route_action"] = result.get("route_action", "retrieve")

        if state["query_domain"] == "out_of_domain":
            state["route_action"] = "answer"
            state["answer"] = "我主要负责育儿知识库相关问题，可以帮你查找育儿建议、亲子案例、沟通话术和知识科普内容。"
        elif state["route_action"] == "answer" and not state.get("answer"):
            state["answer"] = self._fallback_direct_answer(query)

        self.logger.info(f"路由分析: age_group={state['age_group']}, "
                         f"problem_type={state['problem_type']}, "
                         f"query_domain={state['query_domain']}, "
                         f"route_action={state['route_action']}")

        # 思考：路由结果
        task_id = state.get("task_id", "")
        if task_id and state.get("is_stream"):
            age = state["age_group"]
            ptype = state["problem_type"]
            domain = state["query_domain"]
            if domain == "out_of_domain":
                self._push_thinking(task_id, "识别为非育儿问题，直接回答")
            elif state["route_action"] == "answer":
                self._push_thinking(task_id, "识别为简单问候/寒暄，直接回答")
            else:
                self._push_thinking(task_id,
                    f"识别问题意图：{age + ' ' if age else ''}{ptype if ptype else '通用育儿'}问题"
                )

        return state

    @staticmethod
    def _fallback_direct_answer(query: str) -> str:
        normalized = (query or "").strip()
        if any(word in normalized for word in ["你好", "您好", "hello", "hi"]):
            return "你好，我是育儿知识助手。你可以问我孩子情绪管理、亲子沟通、学习习惯、睡眠饮食、入园适应等问题。"
        if any(word in normalized for word in ["谢谢", "感谢", "辛苦"]):
            return "不客气，我随时可以继续帮你查育儿建议、案例或沟通话术。"
        return "我可以帮你基于育儿知识库检索建议、案例、沟通话术和科普内容。请告诉我孩子年龄段和具体场景，我会更容易找到相关资料。"

    def _format_history(self, history: list) -> str:
        if not history:
            return "无历史对话"
        lines = []
        for h in history[-6:]:
            role = "用户" if h.get("role") == "user" else "助手"
            text = h.get("text", "")
            lines.append(f"{role}: {text[:200]}")
        return "\n".join(lines)

    def _fallback_analyze(self, query: str) -> dict:
        """LLM 不可用时的规则 fallback。"""
        age_group = ""
        for label, keywords in self.AGE_PATTERNS.items():
            for kw in keywords:
                if kw in query:
                    age_group = label
                    break
            if age_group:
                break

        problem_type = ""
        problem_keywords = {
            "情绪管理": ["发脾气", "哭闹", "情绪", "崩溃", "生气", "失控"],
            "行为引导": ["不听话", "打人", "规则", "行为", "习惯"],
            "学习能力": ["作业", "学习", "专注", "注意力", "拖延"],
            "亲子沟通": ["沟通", "话术", "怎么跟", "顶嘴", "手机"],
            "睡眠习惯": ["睡眠", "睡觉", "入睡", "作息"],
            "饮食习惯": ["挑食", "吃饭", "餐桌"],
            "手足关系": ["手足", "兄弟", "姐妹", "二胎", "争抢"],
            "入园适应": ["入园", "上学", "分离", "幼儿园"],
        }
        for label, keywords in problem_keywords.items():
            for kw in keywords:
                if kw in query:
                    problem_type = label
                    break
            if problem_type:
                break

        return {
            "rewritten_query": query,
            "age_group": age_group,
            "problem_type": problem_type,
            "query_domain": "parenting",
            "route_action": "retrieve",
        }
