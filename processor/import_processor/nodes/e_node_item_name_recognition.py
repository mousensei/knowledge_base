import json
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType

from config.milvus_config import milvus_config
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.prompt.item_name_recognition import ITEM_NAME_SYSTEM_PROMPT, \
    ITEM_NAME_USER_PROMPT_TEMPLATE
from processor.import_processor.state import ImportGraphState
from utils.embedding_utils import generate_embeddings
from utils.llm_utils import get_llm_client
from utils.milvus_utils import get_milvus_client, escape_milvus_string


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
        # print(context)

        # 步骤3：调用大模型识别商品名称
        item_name = self._step_3_call_llm(file_title, context)
        print(f"主体名称:" + item_name)

        # 步骤4：回填商品名称到状态和切片
        self._step_4_update_chunks(state, chunks, item_name)
        chunks = state.get("chunks")
        # path = Path(state.get("md_path")).parent / f'{state.get("file_title")}_new_chunks.json'
        # with open(path, "w", encoding="utf-8") as f:
        #     json.dump(
        #         chunks,
        #         f,
        #         ensure_ascii=False,
        #         indent=2,
        #     )

        # 步骤5：为商品名称生成稠密/稀疏向量
        dense_vector, sparse_vector = self._step_5_generate_vectors(item_name)
        print(len(dense_vector))

        # 步骤6：将数据存入Milvus向量数据库
        self._step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)

        # 打印识别结果
        self.logger.info(f"--- 识别完成: {item_name} ---")

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
        total_chars: int = 0
        for index, chunk in enumerate(chunks[:k], start=1):
            # 上下文：title + content
            chunk_title = chunk.get("title", "").strip()
            chunk_content = chunk.get("content", "").strip()

            # 模型文本切片
            piece = f"切片{index}\n标题{chunk_title}\n内容{chunk_content}"
            parts.append(piece)

            # 每片长度限制
            total_chars += len(piece)
            if total_chars >= chunk_size:
                break
        # 切片处理
        context = "\n\n".join(parts).strip()
        final_context = context[:chunk_size]

        return final_context

    def _step_3_call_llm(self, file_title: str, context: str):
        print("主体识别节点,_step_3_call_llm...")
        if not context:
            return file_title

        # ai
        llm_ai = get_llm_client()

        # 提示词
        prompt = f"""
        请从以下信息中识别出商品名称与型号：
        文件名：{file_title}

        正文切片（用于辅助识别）：
        {context}

        要求：
        1. 返回内容为字符串形式，最好是带品牌、型号和名称的完整商品名称。比如：苏伯尓5000W大功率电磁炉；
        2. 返回结果应该只包含商品名称，不要添加任何解释或其他内容；
        3. 如果无法识别商品名称,请返回空字符串。
        """
        message = [
            SystemMessage(ITEM_NAME_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        response = llm_ai.invoke(message)
        item_name = getattr(response, "content", "").strip()
        item_name = item_name.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")

        if not item_name:
            item_name = file_title

        return item_name

    def _step_4_update_chunks(self, state: ImportGraphState, chunks: list[str, str], item_name: str):
        if (not chunks) or (not item_name):
            item_name = state.get("file_title")
        for chunk in chunks:
            chunk["item_name"] = item_name
        return state

    def _step_5_generate_vectors(self, item_name):
        embeddings = generate_embeddings([item_name])
        dense = embeddings["dense"][0]
        sparse = embeddings["sparse"][0]
        return dense, sparse

    def _step_6_save_to_milvus(self, state, file_title: str, item_name: str, dense_vector, sparse_vector):
        print("主体名称识别节点,_step_6_save_to_milvus...")
        milvus_url = milvus_config.milvus_url
        item_name_collection = milvus_config.item_name_collection

        if not milvus_url or not item_name_collection:
            raise Exception("配置信息异常...")

        # 1 milvus_client
        client = get_milvus_client()

        # 2 声明字段schema
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 3 声明索引index,两个字段：稠密和稀疏向量
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="IVF_FLAT",  # 检索算法:分组检索
            metric_type="COSINE",  # 相似度函数：余弦相似度
            params={"nlist": 128}
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",  # 检索算法:倒排索引
            metric_type="IP",  # 相似度函数：IP内积值
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",  # 逐个检索取最大值
                "normalize": True,  # 归一化，让IP内积值和COSINE
                "quantization": "none"  # 关闭精度压缩
            }
        )

        # 4 建表
        if not client.has_collection(item_name_collection):
            client.create_collection(collection_name=item_name_collection, schema=schema, index_params=index_params)

        # 5 插入
        safe_item_name = escape_milvus_string(item_name)
        filter_expr = f"item_name=='{safe_item_name}'"  # item_name==‘兄弟烫金机’
        client.delete(collection_name=item_name_collection, filter=filter_expr)  # 幂等性删除
        data = {"file_title": file_title, "item_name": item_name, "dense_vector": dense_vector,
                "sparse_vector": sparse_vector}
        insert_result = client.insert(item_name_collection, [data])
        client.load_collection(item_name_collection)  # 将加入的数据加载到milvus数据库的缓冲区，提高检索速度

        # 6 返回主键
        print(insert_result)

        state["item_name"] = item_name
        return state


if __name__ == "__main__":
    node = NodeItemNameRecognition()
    with open("D:\output\hak180产品安全手册\hak180产品安全手册_chunks.json", "r", encoding="utf-8") as f:
        chunks_content = f.read()

    init_state = {
        "chunks": json.loads(chunks_content),
        "file_title": "hak180产品安全手册",
        "md_path": "D:\output\hak180产品安全手册\hak180产品安全手册_new.md"
    }

    response = node(init_state)
