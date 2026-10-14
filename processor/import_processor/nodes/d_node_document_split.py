import json
import re
from pathlib import Path
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.import_config import get_config
from processor.import_processor.state import ImportGraphState


class NodeDocumentSplit(BaseNode):
    """
    文档切分节点：智能文档切片
    """

    name = "node_document_split"

    def process(self, state: ImportGraphState):
        print("node_document_split...文档切割节点...")

        # 步骤1：加载并标准化输入数据
        content, file_title = self._step_1_get_inputs(state)

        # 步骤2：按MD标题进行初次切分
        sections, title_count, lines_count = self._step_2_split_by_titles(content, file_title)
        print(f"title_count:{title_count}")
        print(f"lines_count:{lines_count}")
        # for section in sections:
        #     print(section)

        # 步骤3：无标题场景兜底处理
        sections = self._step_3_handle_no_title(content, sections, title_count, file_title)

        # 步骤4：Chunk精细化处理（长切短合）
        sections = self._step_4_refine_chunks(sections)
        for section in sections:
            print(section.get("title"))
            print(section.get("part"))

        # 步骤5：输出文档切分统计信息
        self._step_5_print_stats(lines_count, sections)

        # 步骤6：Chunk结果本地JSON备份
        self._step_6_backup(state, sections)

        state["chunks"] = sections
        return state

    # 步骤1参数
    def _step_1_get_inputs(self, state):
        md_content = state.get("md_content")
        if not md_content:
            raise StateFieldError(message="文档不存在...")

        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(message="文档名称不存在...")
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
        return md_content, file_title

    # 步骤2标题初切
    def _step_2_split_by_titles(self, content: str, file_title: str):
        in_code_block = False
        title_count: int = 0
        current_title: str = ""

        lines = content.split("\n")  # 文档的所有行信息
        current_lines = []
        sections: list[dict[str, str]] = []

        # 刷入sections
        def _flush_section():
            if not current_lines:
                return
            sections.append({
                "title": current_title,
                "content": "\n".join(current_lines),
                "parent_title": "",
                "file_title": file_title
            })

        # 循环文档的所有行
        title_pattern = r'\s*#{1,6}\s+.+'
        for line in lines:
            # 处理干净的行
            strip_line = line.strip()

            # 是不是代码块
            if strip_line.startswith("```") or strip_line.startswith("~~~"):
                in_code_block = not in_code_block
                current_lines.append(line)
                continue

            is_title = re.match(title_pattern, line)  # 是不是标题
            if (not in_code_block) and is_title:
                _flush_section()
                current_title = strip_line
                current_lines = [current_title]
                title_count += 1
            else:
                current_lines.append(line)  # 存储普通行或者代码块的内容

        # 循环结束后将最后的所有剩下的行刷入sections
        _flush_section()

        return sections, title_count, len(lines)

    # 步骤3无标题兜底
    def _step_3_handle_no_title(self, content, sections, title_count, file_title):
        if title_count == 0:
            return [{"title": "无标题", "content": content, "file_title": file_title}]
        return sections

    # 步骤4精细化处理(长切短合)
    def _step_4_refine_chunks(self, sections):
        print("切分节点，_step_4_refine_chunks...")

        # 长切
        refined_split = []
        for section in sections:
            refined_split.extend(self._split_long_section(section))

        # 短合
        final_sections = self._merge_short_sections(refined_split)

        # 兜底parent_title
        for section in final_sections:
            if not section.get("parent_title"):
                section["parent_title"] = section.get("title")

        return final_sections

    # 步骤4方法1长切
    def _split_long_section(self, section: dict[str, str]):
        content = section.get("content", "")
        content_len = len(content)

        # 符合标准则直接返回
        if content_len <= get_config().max_content_length:
            return [section]

        # 切分逻辑
        title = section["title"]
        prefix = f"{title}\n\n" if title else ""
        body = content
        body = body[body.find(title) + len(title):].lstrip()  # 切分之前砍掉title
        available_len = get_config().max_content_length - len(prefix)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available_len,
            chunk_overlap=10,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "]
        )

        sub_sections = []
        split_bodys = splitter.split_text(body)
        for index, chunk in enumerate(split_bodys, start=1):
            text = chunk.strip()
            if not text:
                continue
            full_text = (prefix + text).strip()
            sub_sections.append({
                "title": f"{title}-{index}" if title else f"chunk-{index}",
                "content": full_text,
                "parent_title": title,
                "part": index,
                "file_title": section.get("file_title"),
            })
        return sub_sections

    # 步骤4方法2短合
    def _merge_short_sections(self, sections: list[dict[str, str]]):

        if not sections:
            return []

        merged_sections = []
        current_chunk = None  # 当前章节

        # 合并章节
        for sec in sections:
            if current_chunk is None:
                current_chunk = sec
                continue
            # 判断
            is_current_short = len(current_chunk.get("content")) < get_config().min_content_length  # 长度是否小于500
            is_same_parent = current_chunk.get("parent_title") == sec.get("parent_title")
            if is_current_short and is_same_parent:
                # 合并
                parnt_title = sec.get("parnt_title")
                next_content = sec.get("content")
                if parnt_title and next_content.startswith(parnt_title):
                    next_content = next_content[len(parnt_title):].lstrip()
                current_chunk["content"] += "\n\n" + next_content
                if "part" in sec:
                    current_chunk["part"] = sec["part"]  # 将来对章节来源的可追溯性
            else:
                merged_sections.append(current_chunk)
                current_chunk = sec

        if current_chunk is not None:
            merged_sections.append(current_chunk)

        return merged_sections

    def _step_5_print_stats(self, lines_count: int, sections: List[Dict[str, str]]) -> None:
        print("切分节点，打印日志...")
        chunk_num = len(sections)
        # 输出核心统计信息：原始行数/最终Chunk数/首个Chunk预览
        self.logger.info("-" * 50 + " 文档切分统计信息 " + "-" * 50)
        self.logger.info(f"MD原始文本总行数：{lines_count}")
        self.logger.info(f"最终生成Chunk数量：{chunk_num}")

    def _step_6_backup(self, state, sections):
        path = Path(state.get("md_path")).parent / f"{state.get('file_title')}_chunks.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                sections,
                f,
                ensure_ascii=False,  # 保留中文，不转义为\u编码
                indent=2  # 格式化缩进，便于阅读
            )


if __name__ == "__main__":
    with open("D:\output\hak180产品安全手册\hak180产品安全手册_new.md", "r", encoding="utf-8") as f:
        md_content = f.read()
    init_state = {
        "md_path": "D:\output\hak180产品安全手册\hak180产品安全手册_new.md",
        "md_content": md_content,
        "file_title": "hak180产品安全手册",
    }
    node = NodeDocumentSplit()
    response = node(init_state)
