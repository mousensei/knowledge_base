import json

from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search


class NodeSearchEmbedding(NodeBase):
    """
   节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
   """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # TODO
        logger.info(f"【{self.name}】节点逻辑")

        # 1 参数校验
        item_names = state.get("item_names")  # 标量过滤条件
        query = state.get("rewritten_query")  # 搜索相似信息

        # 2 milvus
        query__embeddings = generate_embeddings([query])
        dense_vector = query__embeddings["dense"][0]
        sparse_vector = query__embeddings["sparse"][0]
        milvus_client = get_milvus_client()

        # 3 搜索
        kb_chunks = milvus_config.chunks_collection
        reqs = create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            expr=f'item_name in {item_names}'
        )
        responses = hybrid_search(
            client=milvus_client,
            collection_name=kb_chunks,
            reqs=reqs,
            ranker_weights=(0.8, 0.2),
            output_fields=["chunk_id", "content", "item_name"]
        )
        # print(responses)
        # 4 处理结果
        return {"embedding_chunks": responses[0] if responses else []}


if __name__ == "__main__":
    searchNode = NodeSearchEmbedding()
    init_state = {
        "item_names": ["华为擎云B530台式机", "华为擎云M272Q显示器"],
        "rewritten_query": "华为擎云B530台式机和华为擎云M272Q显示器怎么才能点亮？"
    }
    responses = searchNode.process(init_state)
    print("============================")
    print(json.dumps(responses, ensure_ascii=False, indent=4))
    print("============================")
