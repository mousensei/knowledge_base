from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState


class NodeItemNameRecognition(BaseNode):
    """
    主体识别节点：主体识别与标签提取
    """

    name = "node_item_name_recognition"

    def process(self, state: ImportGraphState):
        print("node_item_name_recognition...主体名称识别节点...")
        # 步骤1：提取并校验输入
        file_title, chunks = self._step_1_get_inputs(state)

        # 步骤2：构建大模型识别的上下文
        context = self._step_2_build_context(chunks)
        print(context)

        # # 步骤3：调用大模型识别商品名称
        # item_name = self._step_3_call_llm(file_title, context)
        #
        # # 步骤4：回填商品名称到状态和切片
        # self._step_4_update_chunks(state, chunks, item_name)
        #
        # # 步骤5：为商品名称生成稠密/稀疏向量
        # dense_vector, sparse_vector = self._step_5_generate_vectors(item_name)
        #
        # # 步骤6：将数据存入Milvus向量数据库
        # self._step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)

        # 打印识别结果
        # self.logger.info(f"--- 识别完成: {item_name} ---")

        return state

    def _step_1_get_inputs(self, state):
        print("主体识别节点，_step_1_get_inputs...")
        # 1、参数校验
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        chunks = state.get("chunks")
        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        return file_title, chunks

    def _step_2_build_context(self, chunks):
        print("主体识别节点，_step_2_build_context...")
        k = self.config.item_name_chunk_k  # 文本块限制
        chunk_size = self.config.item_name_chunk_size  # 文本长度限制

        # 上下内容列表
        parts: list[str] = []
        for index, chunk in enumerate(chunks[:k], start=1):
            # 上下文：title + content
            chunk_title = chunk.get("title", "").strip()
            chunk_content = chunk.get("content", "").strip()

            # 模型文本切片
            piece = f"切片{index}\n标题{chunk_title}\n内容{chunk_content}"
            parts.append(piece)

            # 每片长度限制
            total_chars = len(piece)
            if total_chars >= chunk_size:
                break
        # 切片处理
        context = "\n\n".join(parts).strip()
        final_context = context[:chunk_size]

        return final_context


if __name__ == "__main__":
    node = NodeItemNameRecognition()
