import os
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from pymilvus import MilvusClient
from FlagEmbedding import BGEM3FlagModel

# 关闭 Unstructured 广告提示
os.environ["UNSTRUCTURED_DISABLE_NOTICE"] = "1"


def get_milvus_client():
    client = MilvusClient(
        uri="http://192.168.222.99:19530/",
        token=""
    )
    return client


def insert_data(client: MilvusClient, collection_name: str):
    # 加载文档
    loader = UnstructuredLoader(file_path="sample.docx", mode="single")
    doc = loader.load()

    # 切分文档
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。"]
    )
    doc_list = splitter.split_documents(doc)
    doc_list = doc_list[0:20]
    doc_max_len = max([len(bytes(doc.page_content.encode("utf-8"))) for doc in doc_list])
    print(doc_max_len)

    # 向量化
    bgem3model = BGEM3FlagModel(
        model_name_or_path=r"D:/ai_models/modelscope_cache/models/BAAI/bge-m3",
        return_dense=True,
        return_sparse=True,
    )
    page_contents = [doc.page_content for doc in doc_list]
    all_vectors = bgem3model.encode(
        page_contents,
        return_sparse=True,
        return_dense=True,
    )

    # ========== 【必填】关键修复：显式转换为 float32 ==========
    dense_vecs = np.array(all_vectors['dense_vecs'], dtype=np.float32)
    sparse_vecs = all_vectors['lexical_weights']
    # ==========================================================

    # 准备插入数据
    insert_data_list = []
    for doc, dense, sparse in zip(doc_list, dense_vecs, sparse_vecs):
        insert_data_list.append(
            {
                "vector": dense,  # 此时 dense 已是 float32 格式数组
                "sparse_vector": sparse,
                "text": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    # 插入数据
    count = client.insert(
        collection_name=collection_name,
        data=insert_data_list,
    )
    print(f"成功插入 {count} 条")


if __name__ == '__main__':
    client = get_milvus_client()
    insert_data(client, "demo_collection")
