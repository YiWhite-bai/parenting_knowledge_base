"""Markdown 文档切分节点。"""

import json
import os
import re
from typing import Any, Dict, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    """Markdown 文档切分节点。"""



    name = "document_split_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        config = self.config
        md_content, file_title, max_content_length, min_content_length = self._validate_state(state, config)

        sections: List[Dict[str, Any]] = self._split_by_headings(md_content, file_title)
        final_sections = self._split_and_merge(sections, max_content_length, min_content_length)
        final_chunks = self._assemble_chunks(final_sections)

        # 注入育儿元数据到每个 chunk
        metadata = state.get("_metadata", {})
        for chunk in final_chunks:
            chunk["age_group"] = metadata.get("age_group", "")
            chunk["content_type"] = metadata.get("content_type", "")
            chunk["problem_type"] = metadata.get("problem_type", "")
            chunk["scene"] = metadata.get("scene", "")
            chunk["author"] = metadata.get("author", "")
            chunk["source_file"] = metadata.get("source_file", "")
            chunk["source_path"] = metadata.get("source_path", "")

        self._backup_chunks(final_chunks, state)
        state["chunks"] = final_chunks
        return state

    def _validate_state(self, state: ImportGraphState, config) -> Tuple[str, str, int, int]:
        self.log_step("Step 1", "切分文档的参数校验以及获取...")
        md_content = state.get("md_content")
        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
        file_title = state.get("file_title")

        if config.max_content_length <= 0 or config.min_content_length <= 0 \
                or config.max_content_length <= config.min_content_length:
            raise ValueError("切片长度参数校验失败，请检查 max/min_content_length 配置。")

        return md_content, file_title, config.max_content_length, config.min_content_length

    def _split_by_headings(self, md_content: str, file_title: str) -> List[Dict[str, Any]]:
        self.log_step("Step 2", "根据标题进行切分...")
        in_fence = False
        body_lines = []
        sections = []
        current_title = ""
        hierarchy = [""] * 7
        current_level = 0

        def _flush():
            body = "\n".join(body_lines)
            if current_title or body:
                parent_title = ""
                for i in range(current_level - 1, 0, -1):
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break
                if not parent_title:
                    parent_title = current_title if current_title else file_title
                sections.append({
                    "title": current_title if current_title else file_title,
                    "body": body,
                    "parent_title": parent_title,
                    "file_title": file_title,
                })

        md_lines = md_content.split("\n")
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")
        for md_line in md_lines:
            if md_line.strip().startswith("```") or md_line.strip().startswith("~~~"):
                in_fence = not in_fence

            match = heading_re.match(md_line) if not in_fence else None
            if match:
                _flush()
                current_title = md_line
                level = len(match.group(1))
                current_level = level
                hierarchy[level] = current_title
                for i in range(level + 1, 7):
                    hierarchy[i] = ""
                body_lines = []
            else:
                body_lines.append(md_line)

        _flush()
        return sections

    def _split_and_merge(self, sections: List[Dict[str, Any]], max_content_length: int,
                         min_content_length: int) -> List[Dict[str, Any]]:
        current_sections = []
        for section in sections:
            current_sections.extend(self._split_long_section(section, max_content_length))
        final_sections = self._merge_short_section(current_sections, min_content_length)
        return final_sections

    def _split_long_section(self, section: Dict[str, Any], max_content_length: int) -> List[Dict[str, Any]]:
        body = section.get("body", "")
        title = section.get("title", "")
        parent_title = section.get("parent_title", "")
        file_title = section.get("file_title", "")

        if len(title) > 50:
            title = title[:50]

        if "<table>" in body:
            self.logger.info("检测到 section 中有 HTML 表格，执行表格降维处理")
            body = MarkdownTableLinearizer.process(body)
            section["body"] = body

        title_prefix = f"{title}\n\n"
        total_length = len(title_prefix) + len(body)
        if total_length <= max_content_length:
            return [section]

        body_length = max_content_length - len(title_prefix)
        if body_length <= 0:
            return [section]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_length,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
            keep_separator=True,
        )
        texts = text_splitter.split_text(body)
        if len(texts) == 1:
            return [section]

        sub_sections = []
        for index, text_chunk in enumerate(texts):
            sub_sections.append({
                "body": text_chunk,
                "title": f"{title}-{index + 1}",
                "parent_title": parent_title,
                "file_title": file_title,
            })
        return sub_sections

    def _merge_short_section(self, current_sections: List[Dict[str, Any]], min_content_length: int) -> List[Dict[str, Any]]:
        if not current_sections:
            return []
        current_section = current_sections[0]
        final_sections = []
        for next_section in current_sections[1:]:
            same_parent = current_section.get("parent_title") == next_section.get("parent_title")
            if same_parent and len(current_section.get("body", "")) < min_content_length:
                current_section["body"] = (
                    current_section.get("body", "").rstrip() + "\n\n" + next_section.get("body", "").lstrip()
                )
                current_section["title"] = current_section["parent_title"]
            else:
                final_sections.append(current_section)
                current_section = next_section
        final_sections.append(current_section)
        return final_sections

    def _assemble_chunks(self, final_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        final_chunks = []
        for section in final_sections:
            body = section.get("body", "")
            title = section.get("title", "")
            parent_title = section.get("parent_title", "")
            file_title = section.get("file_title", "")
            content = f"{title}\n\n{body}"
            final_chunks.append({
                "content": content,
                "title": title,
                "parent_title": parent_title,
                "file_title": file_title,
            })
        self.logger.info(f"最终切割后能够进入到嵌入节点的 chunk 个数: {len(final_chunks)}")
        return final_chunks

    def _backup_chunks(self, final_chunks: List[Dict[str, Any]], state: ImportGraphState):
        local_dir = state.get("file_dir", "")
        if not local_dir:
            return
        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_chunks, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"备份 chunks 失败: {e}")
