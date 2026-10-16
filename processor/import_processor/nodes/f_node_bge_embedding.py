import json

from processor.import_processor.base import BaseNode
from processor.import_processor.state import ImportGraphState
from utils.embedding_utils import generate_embeddings


class NodeBGEEmbedding(BaseNode):
    """
    混合向量化节点：使用 BGE-M3 模型将文本转换为向量
    """

    name = "node_bge_embedding"

    def process(self, state: ImportGraphState):
        print("node_bge_embedding...向量化节点...")

        # 1 参数校验
        chunks = self._step_1_validate_paths(state)

        # 2 向量化chunk数据
        output_data = self._step_2_generate_embeddings(chunks)

        # 3 更新state
        state["chunks"] = output_data
        return state

    def _step_1_validate_paths(self, state: ImportGraphState):
        print("向量化节点,_step_1_validate_paths...")
        chunks = state.get("chunks")

        if not chunks:
            raise Exception(f"参数错误：{chunks}")

        if not isinstance(chunks, list):
            raise Exception("chunks参数类型错误")
        return chunks

    def _step_2_generate_embeddings(self, chunks: list[str, str]):
        print("向量化节点,_step_2_generate_embeddings...")

        output_data = []
        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            five_ready_embed_texts = []  # 5条准备向量化的数据
            five_texts = chunks[i:i + batch_size]
            for five_text in five_texts:
                item_name = five_text["item_name"]
                content = five_text["content"]
                five_ready_embed_text = f"{item_name}\n{content}" if item_name else content
                five_ready_embed_texts.append(five_ready_embed_text)
            # 向量化(content)
            embeddings = generate_embeddings(five_ready_embed_texts)
            print(f"embeddings:{embeddings}")
            # 结果回填
            for j, doc in enumerate(five_texts):
                item = doc.copy()
                dense = embeddings["dense"][j]
                item["dense_vector"] = dense
                sparse = embeddings["sparse"][j]
                item["sparse_vector"] = sparse
                output_data.append(item)

        return output_data
