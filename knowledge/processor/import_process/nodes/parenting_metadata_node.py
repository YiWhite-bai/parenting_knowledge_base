"""育儿元数据提取节点。

职责：使用 LLM 从育儿文档内容中提取结构化元数据：
标题、年龄段、内容类型、问题类型、场景描述。
并读取 Markdown 内容，存入 state 供后续切分使用。
"""

from pathlib import Path
import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import LLMError, StateFieldError
from knowledge.prompts.import_prompt import (
    PARENTING_METADATA_SYSTEM_PROMPT,
    PARENTING_METADATA_USER_PROMPT_TEMPLATE,
)
from knowledge.utils.client.ai_clients import AIClients


class ParentingMetadataNode(BaseNode):
    """育儿元数据提取节点。

    从育儿文档 Markdown 内容中提取标题、年龄段、内容类型、问题类型、场景描述。
    """

    name = "parenting_metadata_node"

    # 已知的分类目录到内容类型的映射（优先使用 LLM 推断，此映射为 fallback）
    CATEGORY_TO_CONTENT_TYPE = {
        "育儿建议": "育儿建议",
        "专家建议": "专家文章",
        "亲子案例": "亲子案例",
        "沟通话术": "沟通话术",
        "知识科普": "知识科普",
    }

    # 年龄段关键词匹配（LLM 推断的 fallback）
    AGE_PATTERNS = [
        (re.compile(r"0[-\s]*3\s*岁|婴幼儿|依恋|如厕训练|分离焦虑|咿呀"), "0-3岁"),
        (re.compile(r"3[-\s]*6\s*岁|学龄前|幼儿园|入园|睡前拖延|挑食"), "3-6岁"),
        (re.compile(r"6[-\s]*12\s*岁|学龄|小学|作业拖延|同伴|专注力"), "6-12岁"),
        (re.compile(r"12[+\s]*(岁以上|岁)|青春期|顶嘴|独立|手机使用|社交媒体"), "12+岁"),
    ]

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("Step 1", "读取 Markdown 内容")
        md_path = state.get("md_path", "")
        if md_path:
            content = Path(md_path).read_text(encoding="utf-8").strip()
        else:
            raise StateFieldError(
                node_name=self.name, field_name="md_path",
                message="未找到 Markdown 文件路径"
            )

        if not content:
            self.logger.warning("Markdown 内容为空")
            state["md_content"] = ""
            return state

        # 统一换行
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        state["md_content"] = content

        self.log_step("Step 2", "使用 LLM 提取育儿元数据")
        file_title = state.get("file_title", "")
        source_category = state.get("source_category", "")

        # 取内容前 3000 字符供 LLM 分析
        preview = content[:3000]

        try:
            llm = AIClients.get_llm_client(response_format=True)
            user_prompt = PARENTING_METADATA_USER_PROMPT_TEMPLATE.format(
                file_title=file_title,
                source_category=source_category,
                context=preview,
            )
            messages = [
                SystemMessage(content=PARENTING_METADATA_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            response = llm.invoke(messages)
            metadata = self._parse_llm_json(response.content)
        except Exception as e:
            self.logger.warning(f"LLM 元数据提取失败，使用 fallback: {e}")
            metadata = self._fallback_extract(file_title, content, source_category)

        # 将元数据写入 state（LangGraph 的 TypedDict 可能不支持动态字段，我们存到 chunks 的元数据里）
        # 这里先暂存，后续在 embedding 节点注入到每个 chunk
        state["file_title"] = metadata.get("title", file_title)
        state["_metadata"] = {
            "title": metadata.get("title", file_title),
            "author": metadata.get("author", ""),
            "age_group": metadata.get("age_group", ""),
            "content_type": metadata.get("content_type", self.CATEGORY_TO_CONTENT_TYPE.get(source_category, "知识科普")),
            "problem_type": metadata.get("problem_type", "育儿建议"),
            "scene": metadata.get("scene", file_title),
            "source_file": state.get("source_file", ""),
            "source_path": state.get("source_path", ""),
        }

        self.logger.info(f"提取元数据: {json.dumps(state['_metadata'], ensure_ascii=False)}")
        return state

    @staticmethod
    def _parse_llm_json(content: str) -> dict:
        """解析 LLM JSON 输出，兼容 ```json 围栏。"""
        cleaned = str(content or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("LLM 元数据结果不是 JSON 对象")
        return parsed

    def _fallback_extract(self, file_title: str, content: str, source_category: str) -> dict:
        """LLM 不可用时的规则 fallback。"""
        # 年龄段
        age_group = ""
        for pattern, label in self.AGE_PATTERNS:
            if pattern.search(file_title) or pattern.search(content[:500]):
                age_group = label
                break

        # 内容类型
        content_type = self.CATEGORY_TO_CONTENT_TYPE.get(source_category, "知识科普")

        # 问题类型
        problem_type_map = {
            "育儿建议": "育儿建议",
            "专家建议": "专家建议",
            "亲子案例": "案例分析",
            "沟通话术": "亲子沟通",
            "知识科普": "知识科普",
        }
        problem_type = problem_type_map.get(source_category, "育儿建议")

        author = ""
        author_match = re.search(r"(?:作者|专家|来源|机构)\s*[:：]\s*([^\n\r]+)", content[:1000])
        if author_match:
            author = author_match.group(1).strip()[:50]

        return {
            "title": file_title,
            "author": author,
            "age_group": age_group,
            "content_type": content_type,
            "problem_type": problem_type,
            "scene": file_title,
        }
