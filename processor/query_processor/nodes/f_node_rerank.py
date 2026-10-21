from typing import List, Dict, Any

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.reranker_http_utils import rerank_documents


class NodeRerank(NodeBase):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rerank"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑：将两路向量结果和mcp网络结果重新定义结构和模型排序
        """
        # TODO
        logger.info(f"【{self.name}】节点逻辑,重排序...")

        # 1 参数重定义数据结构
        merged_multi_docs: List[Dict[str, Any]] = self._step_1_merge_multi_source_docs(state)
        # for doc in merged_multi_docs:
        #     print(doc)

        # 2 调用重排序模型
        reranked_docs = self._step_2_rerank_merged_docs(state, merged_multi_docs)

        # 3 断崖测试保留top_k条数据
        cutoff_docs = self._step_3_cliff_cutoff(reranked_docs)
        for doc in cutoff_docs:
            print(doc)

        # 4 更新state
        state["reranked_docs"] = cutoff_docs
        return state

    # 步骤1参数数据结构处理
    def _step_1_merge_multi_source_docs(self, state):
        final_docs = []  # content,title,chunk_id,url,source
        rrf_chunks = state["rrf_chunks"]
        web_docs = state["web_search_docs"]

        # 处理rrf_chunks结构
        for chunk in rrf_chunks:
            format_rrf_doc = {
                "content": chunk.get("content"),
                "title": chunk.get("item_name"),
                "chunk_id": chunk.get("chunk_id"),
                "url": None,
                "source": "local",
            }
            final_docs.append(format_rrf_doc)

        # 处理web_docs结构
        for web_doc in web_docs:
            format_web_doc = {
                "content": web_doc.get("snippet"),
                "title": web_doc.get("title"),
                "chunk_id": None,
                "url": web_doc.get("url"),
                "source": "web",
            }
            final_docs.append(format_web_doc)
        return final_docs

    # 步骤2模型重排序
    def _step_2_rerank_merged_docs(self, state: QueryGraphState, merged_multi_docs: List[Dict[str, Any]]):
        rewritten_query = state.get("rewritten_query")
        contents = [doc.get("content") for doc in merged_multi_docs]
        # 重排序模型得分
        rerank_scores = rerank_documents(rewritten_query, contents)
        # 结果合并
        scored_docs = [{**doc, "score": score} for doc, score in zip(merged_multi_docs, rerank_scores)]
        # for doc, score in zip(merged_multi_docs, rerank_scores):
        #     scored_docs.append({
        #         "content": doc.get("content"),
        #         "title": doc.get("title"),
        #         "chunk_id": doc.get("chunk_id"),
        #         "url": doc.get("url"),
        #         "source": doc.get("source"),
        #         "score": score
        #     })

        # 结果按照匹配分数排序
        sorted_score_docs = sorted(scored_docs, key=lambda x: x["score"], reverse=True)

        return sorted_score_docs

    # 断崖测试，取值
    def _step_3_cliff_cutoff(self, reranked_docs: list[Dict]):
        upper_bound = min(10, len(reranked_docs))  # 断崖结束
        lower_bound = min(3, len(reranked_docs))  # 断崖开始

        cutoff_pos = upper_bound
        for index in range(lower_bound - 1, upper_bound - 1):
            print(index)
            current_score = reranked_docs[index].get("score")
            next_score = reranked_docs[index + 1].get("score")
            # 分差
            abs_gap = current_score - next_score
            # 比率
            rel_gap = abs_gap / (abs(current_score) + 1e-6)
            if abs_gap > 0.5 or rel_gap >= 0.25:
                cutoff_pos = index + 1
                break
        return reranked_docs[:cutoff_pos]


if __name__ == "__main__":
    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {
                "chunk_id": "local_1",
                "item_name": "主板维修手册",
                "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"
            },
            {
                "chunk_id": "local_2",
                "item_name": "闲聊",
                "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"
            },
            {
                "chunk_id": "local_1",
                "item_name": "主板维修手册",
                "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"
            },
            {
                "chunk_id": "local_2",
                "item_name": "闲聊",
                "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"
            },
            {
                "chunk_id": "local_1",
                "item_name": "主板维修手册",
                "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"
            },
            {
                "chunk_id": "local_2",
                "item_name": "闲聊",
                "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"
            },
        ],
        "web_search_docs": [
            {
                "url": "https://example.com/repair",
                "title": "短路查修指南",
                "snippet": "主板通电前先打各主供电电感对地阻值，阻值偏低就是短路。"
            },
            {
                "url": "https://example.com/news",
                "title": "科技新闻",
                "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"
            },
            {
                "url": "https://example.com/repair",
                "title": "短路查修指南",
                "snippet": "主板通电前先打各主供电电感对地阻值，阻值偏低就是短路。"
            },
            {
                "url": "https://example.com/news",
                "title": "科技新闻",
                "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"
            },
            {
                "url": "https://example.com/repair",
                "title": "短路查修指南",
                "snippet": "主板通电前先打各主供电电感对地阻值，阻值偏低就是短路。"
            },
            {
                "url": "https://example.com/news",
                "title": "科技新闻",
                "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"
            },
        ],
    }

    node_rerank = NodeRerank()
    result = node_rerank(mock_state)
