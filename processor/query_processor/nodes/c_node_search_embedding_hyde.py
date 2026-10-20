from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.search_embedding_hyde import HYDE_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.llm_utils import get_llm_client
from utils.milvus_utils import create_hybrid_search_requests, hybrid_search, get_milvus_client


class NodeSearchEmbeddingHyde(NodeBase):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding_hyde"

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
        print(item_names)
        rewritten_query = state.get("rewritten_query")  # 搜索相似信息

        # 2 大模型生成假设文档
        hyde_doc = self._step_2_create_embedding_hyde_doc(rewritten_query)  # 模型扩展结果
        print(hyde_doc)

        # 3 用"重写问题+假设性文档"搜索chunks
        responses = self._step_3_search_embedding_hyde(
            rewritten_query=rewritten_query,
            hyde_doc=hyde_doc,
            item_names=item_names,
        )
        print(responses)
        # 4 结果解析
        return {"hyde_embedding_chunks": responses[0] if responses else []}

    # 步骤2生成假设文档
    def _step_2_create_embedding_hyde_doc(self, rewritten_query: str):
        print("假设性节点：_step_2_create_embedding_hyde_doc...")
        # ai
        ai_client = get_llm_client()

        # 提示词
        prompt_format = HYDE_PROMPT.format(rewritten_query=rewritten_query)

        # 调用模型
        hyde_doc = ai_client.invoke(prompt_format).content

        # 返回结果
        return hyde_doc

    def _step_3_search_embedding_hyde(self, rewritten_query: str, hyde_doc: str, item_names: list[str]):
        # 拼接问题
        embedding_context = rewritten_query + " " + hyde_doc

        # 问题向量化
        embeddings = generate_embeddings([embedding_context])
        dense_vector = embeddings.get("dense")[0]
        sparse_vector = embeddings.get("sparse")[0]

        # 搜索
        milvus_client = get_milvus_client()
        collection_name = milvus_config.chunks_collection
        annRequests = create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            expr=f'item_name in {item_names}',
        )
        responses = hybrid_search(
            client=milvus_client,
            collection_name=collection_name,
            reqs=annRequests,
            limit=5,
            output_fields=["chunk_id", "content", "item_name"]
        )
        return responses


if __name__ == "__main__":
    node_search_embedding_hyde = NodeSearchEmbeddingHyde()
    init_state = {
        "item_names": ["BrotherHAK180烫金机"],
        "rewritten_query": "兄弟请帮我查一下HAK180烫金机是什么?"
    }
    process = node_search_embedding_hyde.process(init_state)
