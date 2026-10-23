from typing import Tuple

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger


class NodeRrf(NodeBase):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web）进行加权融合排序。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        融合节点逻辑
        """
        # TODO
        logger.info(f"【{self.name}】节点逻辑，将向量节点和假设性节点结果融合到一起...")

        # 1 参数融合
        embedding_chunks = state.get("embedding_chunks")
        hyde_embedding_chunks = state.get("hyde_embedding_chunks")

        embedding_chunks_list = [doc.get("entity") for doc in embedding_chunks]  # 5个eneity
        hyde_embedding_chunks_list = [doc.get("entity") for doc in hyde_embedding_chunks]  # 5个eneity

        # 2 构建inputs
        rrf_inputs = [
            (embedding_chunks_list, 1.0),
            (hyde_embedding_chunks_list, 1.0)
        ]

        # 3 融合函数
        rrf_merge_results = self._rrf_merge(rrf_inputs)
        # for rrf_merge_result in rrf_merge_results:
        #     print(rrf_merge_result)

        # 4 提取rrf_chunks（丢弃分数）
        rrf_chunks = [entity for entity, _ in rrf_merge_results]

        # 5 更新返回state
        state["rrf_chunks"] = rrf_chunks
        # print(state["rrf_chunks"])
        return state

    def _rrf_merge(self, rrf_inputs: list[Tuple], k: int = 60, max_results: int = 5):
        print("融合节点，_rrf_merge...")
        chunk_scores = {}  # 融合分{"chunk_id",0.0}
        chunk_data = {}  # 融合数据{"chunk_id",entity}

        # 循环参数列表
        for rrf_input, weight in rrf_inputs:
            for rank, entity in enumerate(rrf_input):
                chunk_id = entity.get("chunk_id")
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight / (k + rank)
                chunk_data.setdefault(chunk_id, entity)
        # 未排序结果
        unsorted_results = [(chunk_data.get(c_id), score) for c_id, score in chunk_scores.items()]

        # 按照rrf融合分排序后的结果
        sorted_results = sorted(
            unsorted_results,
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_results[:max_results] if max_results else sorted_results


if __name__ == "__main__":
    # 模拟两路检索结果
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_5", "content": "向量搜索结果#5"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#4"}},
            {"entity": {"chunk_id": "chunk_6", "content": "HyDE搜索结果#6"}},
        ]
    }

    node_rrf = NodeRrf()
    result = node_rrf(mock_state)
