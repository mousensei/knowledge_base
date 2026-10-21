import json
from os.path import join
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage

from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.item_name_confirm import ITEM_NAME_EXTRACT_TEMPLATE, \
    ITEM_NAME_EXTRACT_SYSTEM_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.llm_utils import get_llm_client
from utils.milvus_utils import hybrid_search, create_hybrid_search_requests, get_milvus_client
from utils.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names


class NodeItemNameConfirm(NodeBase):
    """
    节点功能：确认用户问题中的核心商品名称。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_item_name_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        """
        logger.info(f"【{self.name}】节点逻辑")

        # 1 参数校验
        session_id, original_query = self._step_1_validate_params(state)

        # 2 保存用户消息
        history = get_recent_messages(session_id)
        # print(history)

        # 3 获取历史消息
        message_id = save_chat_message(session_id, "user", original_query)
        print(f"mongo的数据主键{message_id}")

        # 4 llm识别item_names和rewritten_query
        extract_res = self._step_4_extract_info(original_query, history)
        item_names = extract_res["item_names"]
        rewritten_query = extract_res["rewritten_query"]
        state["item_names"] = item_names
        state["rewritten_query"] = rewritten_query

        # 5 向量化检索(item_names)
        query_results = self._step_5_vectorize_and_query(item_names)
        # for item in query_results:
        #     print(item)

        # 6 检索结果对齐
        align_result = self._step_6_align_item_names(query_results)
        print(align_result)
        # 7 状态更新:answer有内容，answer没内容
        state = self._step_7_check_confirmation(state, align_result, history)
        print(state)

        # 8 写入历史消息
        self._step_8_write_history(state, session_id=session_id, rewritten_query=rewritten_query, message_id=message_id)

        return state

    # 步骤1参数校验
    def _step_1_validate_params(self, state):
        session_id = state.get("session_id")
        if not session_id:
            raise Exception("session_id获取失败...")
        original_query = state.get("original_query")
        if not original_query:
            raise ValueError("original_query参数不存在...")

        return session_id, original_query

    # 步骤4提取问题中的设备名称
    def _step_4_extract_info(self, original_query: str, history: list[str, Any]):

        # ai模型
        ai_client = get_llm_client(json_mode=True)

        # 上下文：根据history拼接
        history_text = ""
        for his in history:
            role = his.get("role")
            content = his.get("text")
            history_text += f"{role}: {content}\n"

        # 提示词
        prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(history_text=history_text, query=original_query)
        # print(prompt)
        messages = [
            SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        # 调用模型
        response = ai_client.invoke(messages)
        response_content = response.content

        # 处理结果
        result = json.loads(response_content)
        if "item_names" not in result:
            result["item_names"] = []
        if "rewritten_query" not in result:
            result["rewritten_query"] = original_query
        if result["item_names"]:
            result["item_names"] = [
                name.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")
                for name in result["item_names"]
            ]

        return result

    # 步骤5对item_name向量检索
    def _step_5_vectorize_and_query(self, item_names: list[str]) -> list[Dict]:
        results: list[Dict] = []

        # 1 向量化item_name
        embeddings = generate_embeddings(item_names)

        # 2 检索向量库
        milvus_client = get_milvus_client()
        collection_name = milvus_config.item_name_collection
        for i in range(len(item_names)):
            dense_vector = embeddings.get("dense")[i]
            sparse_vector = embeddings.get("sparse")[i]

            reqs = create_hybrid_search_requests(dense_vector=dense_vector, sparse_vector=sparse_vector, limit=3)
            # 混合检索
            search_res = hybrid_search(
                client=milvus_client,
                collection_name="kb_item_names",
                reqs=reqs,
                ranker_weights=(0.8, 0.2),
                norm_score=True,
                limit=3,
                output_fields=["item_name"]
            )

            # 3 处理检索结果
            matches = []
            if search_res and len(search_res) > 0:
                for hit in search_res[0]:
                    matches.append({
                        "item_name": hit.entity.get("item_name"),
                        "score": hit.get("distance"),
                    })

                results.append({
                    "extracted_name": item_names[i],
                    "matches": matches,
                })
        return results

    # 步骤6结果对齐
    def _step_6_align_item_names(self, query_results: List[Dict]):
        confirmed_item_names: List[str] = []  # 命中选项
        options: List[str] = []  # 可能选项

        # 循环结果，挑选命中项
        for res in query_results:
            extracted_name = (res.get("extracted_name", "") or "").strip()
            matches = res.get("matches", []) or []
            if not matches:
                continue
            # 将matches中的不同分段的数据拿出来
            high_results = [m for m in matches if m.get("score", 0) >= 0.8]  # 0.8高分结果
            mid_results = [m for m in matches if 0.8 > m.get("score", 0) >= 0.6]  # 0.8高分结果

            # 规则A：有0.8以上的，就取0.8分
            if len(high_results) == 1:
                confirmed_item_names.append(high_results[0].get("item_name"))
                continue
            if len(high_results) > 1:
                picked = None
                if extracted_name:
                    for hr in high_results:
                        if hr.get("item_name") == extracted_name:
                            picked = hr
                            break
                if not picked:
                    picked = high_results[0]
                confirmed_item_names.append(picked.get("item_name"))
                continue
            # 规则B：0.8分一下，0.6分以上
            if len(mid_results) > 0:
                for mr in mid_results[:3]:
                    options.append(mr.get("item_name"))
        # 返回处理好的对齐结果
        return {
            "confirmed_item_names": confirmed_item_names,
            "options": options,
        }

    # 更新state状态，确定输出结果（处理命中和options生成answer）
    def _step_7_check_confirmation(self, state: QueryGraphState, align_result: Dict, history: List[Dict]):
        print("处理answer或者item_names")
        confirmed_item_names = align_result.get("confirmed_item_names")
        options = align_result.get("options")

        # 1 有命中(confirmed)
        if confirmed_item_names:
            ids_to_update = []
            for msg in history:
                if not msg.get("item_names"):
                    msgid = msg.get("_id")
                    if msgid:
                        ids_to_update.append(msgid)  # 需要更新的会话消息的id列表
            if ids_to_update:
                update_message_item_names(ids_to_update, confirmed_item_names)
            state["item_names"] = confirmed_item_names
            state["answer"] = ""
            return state

        # 2 有可能选项(options)
        if options:
            state["item_names"] = []
            options_str = "、".join(options) if options else ""
            state["answer"] = f"您是要询问关于以下的产品信息吗:{options_str}"
            return state

        # 3 没有任何命中
        state["answer"] = "抱歉，未找到相关产品，请提供准确型号以便我为您查询。"
        state["item_names"] = []
        return state

    def _step_8_write_history(self, state, session_id: str, rewritten_query: str, message_id: str):

        # 如有answer，插入一条ai消息message
        if state["answer"]:
            save_chat_message(
                session_id=session_id,
                role="assistant",
                text=state["answer"],
                rewritten_query="",
                item_names=state.get("item_names", [])
            )

        # 更新用户消息，将新生成的内容更新的用户消息中
        save_chat_message(
            session_id=session_id,
            role="user",
            text=state["original_query"],
            rewritten_query=rewritten_query,  # 更新重写后的问题
            item_names=state.get("item_names", []),
            message_id=message_id
        )


if __name__ == "__main__":
    # 初始化图状态
    init_state = {
        "original_query": "那个，兄弟，这个网桥玩意怎么才能鼓捣亮了呀？",
        "session_id": "123"  # 前端传递
    }

    # 创建节点对象
    node_item_name_confirm = NodeItemNameConfirm()
    # 执行节点的单元测试
    result = node_item_name_confirm(init_state)
    # 将返回的图状态进行json序列化
    json_state = json.dumps(result, ensure_ascii=False, indent=4)
    # 输出
    # logger.info(json_state)
