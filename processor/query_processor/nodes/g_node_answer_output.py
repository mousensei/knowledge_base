import re
from typing import Dict, List, Tuple, final

from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.answer_prompt import ANSWER_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.llm_utils import get_llm_client
from utils.mongo_history_utils import save_chat_message
from utils.sse_utils import push_to_session, SSEEvent
from utils.task_utils import set_task_result, add_done_task

MAX_CONTENT_CHARS = 12000


class NodeAnswerOutput(NodeBase):
    """
    节点功能: 答案生成
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_answer_output"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        """
        # TODO
        logger.info(f"【{self.name}】节点逻辑")

        # 1 参数：判断是否有answer
        answer_exists = self._step_1_check_answer(state)

        if not answer_exists:
            # 2 如果没answer，构建prompt
            prompt = self._step_2_construct_prompt(state)
            state["prompt"] = prompt
            # 3 调用llm，模型生成答案
            result = self._step_3_generate_response(state, prompt)
            # print("============llm_answer=============")
            # print(result.get("answer"))
            # print("============llm_answer=============")

        # 4 提取图片信息(history and 前端展示)
        image_urls = self._step_4_extract_images_from_docs(state.get("reranked_docs") or [])
        # print("============image_urls=============")
        # for image_url in image_urls:
        #     print(image_url)
        # print("============image_urls=============")

        # 5 把答案写入history(mongo)中
        if state.get("answer"):
            self._step_4_write_history(state, image_urls=image_urls)
        add_done_task(state.get("session_id"), self.name, state.get("is_stream"))  # 放入节点进度

        # 6 流式输出final事件（兜底结果,打印图片）
        if state.get("is_stream"):
            push_to_session(
                state.get("session_id"),
                SSEEvent.FINAL,
                {
                    "answer": state["answer"],
                    "status": "completed",
                    "image_urls": image_urls,
                }
            )

        return state

    # 步骤1参数校验
    def _step_1_check_answer(self, state: QueryGraphState):
        print("输出节点,_tep_1_check_answer...")
        answer = state.get("answer")
        is_stream = state.get("is_stream")

        if answer:
            print("输出节点,系统短路，流式输出结果...")
            if is_stream:
                push_to_session(state.get("session_id"), SSEEvent.DELTA, {"delta": answer})  # sse
            else:
                set_task_result(state.get("session_id"), "answer", answer)  # 任务进度
            return True
        else:
            return False

    # 步骤2提示词
    def _step_2_construct_prompt(self, state: QueryGraphState):
        char_budget = MAX_CONTENT_CHARS  # 提示词字符上限
        # 拼接提示词参数
        question = state.get("rewritten_query") or state.get("original_query", "")
        item_names = state.get("item_names")
        # 提示词：context
        context_str, char_budget = self._format_reranked_docs(state.get("reranked_docs") or [], char_budget)
        # 提示词：history
        history_str, char_budget = self._format_chat_history(state.get("history") or [], char_budget)
        # 提示词：item_names
        item_names_str = ", ".join(item_names) if item_names else "无选定商品"
        # 拼接prompt
        prompt = ANSWER_PROMPT.format(
            context=context_str or "无参考内容",
            history=history_str if history_str else "无历史消息",
            item_names=item_names_str,
            question=question,
        )
        return prompt

    # 步骤2方法1格式化docs(=context)
    def _format_reranked_docs(self, reranked_docs: List[Dict], char_budget: int) -> Tuple[str, int]:
        """格式化重排序文档，带字符预算控制"""
        formatted_lines = []
        used_chars = 0

        # 从重排内容中，提取为资料字符串，不可超过限额
        # 优先使用结构化 reranked_docs（包含 source/chunk_id/url/score），便于约束与引用
        # ---------------------------------------------------------
        # 逻辑解释：
        # 1. 遍历重排序后的文档列表 (reranked_docs)，这些文档已经按相关性从高到低排序。
        # 2. 对每个文档提取关键信息 (text, source, chunk_id, url, title, score)。
        # 3. 构造 "元数据头 + 正文" 格式的字符串，例如：
        #    "[1] [local] [chunk_id=123] [score=0.95] [title=操作手册]
        #     这里是文档的正文内容..."
        # 4. 累加字符长度，如果超过 MAX_CONTEXT_CHARS (如 12000 字符)，则停止添加，
        #    确保 Prompt 长度在 LLM 的处理范围内，避免 Token 溢出。
        # ---------------------------------------------------------
        for idx, doc in enumerate(reranked_docs, start=1):
            content = doc.get("content")
            meta_tags = [f"[{idx}]"]
            for field, template in [
                ("source", "[source={}]"),
                ("chunk_id", "[chunk_id={}]"),
                ("url", "[url={}]"),
                ("title", "[title={}]"),
            ]:
                field_value = doc.get(field)
                if field_value not in (None, ""):
                    meta_tags.append(template.format(str(field_value).strip()))

            relevance_score = doc.get("score")
            if relevance_score is not None:
                meta_tags.append(f"[score={float(relevance_score):.4f}]")

            doc_entry = " ".join(meta_tags) + "\n" + content

            if used_chars + len(doc_entry) > char_budget:
                break

            formatted_lines.append(doc_entry)
            used_chars += len(doc_entry) + 2

        return "\n\n".join(formatted_lines), char_budget - used_chars

    # 步骤2方法1格式化history(=history)
    def _format_chat_history(self, chat_history: List[Dict], char_budget: int) -> Tuple[str, int]:
        """格式化历史对话"""
        formatted_lines = []
        used_chars = 0

        role_label_map = {"user": "用户", "assistant": "助手"}

        for message in chat_history:
            role = message.get("role", "")
            text = message.get("text", "")
            if not text or role not in role_label_map:
                continue

            formatted_line = f"{role_label_map[role]}: {text}"
            used_chars += len(formatted_line) + 1

            if used_chars > char_budget:
                return "\n".join(formatted_lines), char_budget - used_chars

            formatted_lines.append(formatted_line)

        return "\n".join(formatted_lines), char_budget - used_chars

    # 步骤3模型生成答案
    def _step_3_generate_response(self, state: QueryGraphState, prompt: str):
        session_id = state.get("session_id")
        is_stream = state.get("is_stream")
        ai_llm = get_llm_client()

        if is_stream:
            final_text = ""
            try:
                for chunk in ai_llm.stream(prompt):
                    delta = getattr(chunk, "content", "") or ""
                    if delta:
                        final_text += delta
                        push_to_session(session_id, SSEEvent.DELTA, {"delta": delta})
            except Exception as e:
                print(f"输出节点异常：_step_3_generate_response...,{e}")
                push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
            state["answer"] = final_text
        else:
            # 非流式直接调用
            logger.info(f"模式: 非流式输出 (Blocking), Session: {session_id}")
            try:
                response = ai_llm.invoke(prompt)
                content = response.content
                state["answer"] = content
                set_task_result(session_id, "answer", content)
                logger.info(f"生成回答完成，长度: {len(content)}")
            except Exception as e:
                logger.error(f"生成回答出错: {e}", exc_info=True)
                state["answer"] = "抱歉，生成回答时出现错误。"
        return state

    def _step_4_extract_images_from_docs(self, reranked_docs: list[Dict]):
        # 声明
        images = []
        seen = set()
        if not reranked_docs:
            return []

        # web_doc的图片
        for i, doc in enumerate(reranked_docs):
            url = (doc.get("url") or "").strip()
            if url:
                if url.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg")):
                    if url not in seen:
                        seen.add(url)
                        images.append(url)

            # md_content的图片
            # md_img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
            md_img_pattern = re.compile(r'!\[.*?\]\((.*?\.(?:png|jpg|jpeg|gif|bmp|webp|svg))\)')
            text = (doc.get("content") or "").strip()
            if text:
                matches = md_img_pattern.findall(text)
                for match in matches:
                    img_url = match.strip()
                    if img_url and img_url not in seen:
                        seen.add(img_url)
                        images.append(img_url)
        return images

    # 步骤“4写入记忆历史”数据
    def _step_4_write_history(self, state: QueryGraphState, image_urls: list[str]):
        session_id = state.get("session_id", "default")
        answer = (state.get("answer") or "").strip()
        item_names = state.get("item_names") or []
        print("answer:", answer)
        if answer:
            try:
                save_chat_message(
                    session_id=session_id,
                    role="assistant",
                    text=answer,
                    rewritten_query="",
                    item_names=item_names,
                    image_urls=image_urls,
                    message_id=None
                )
            except Exception as e:
                print(f"导入mongodb失败，失败原因:{str(e)}")
        return state
