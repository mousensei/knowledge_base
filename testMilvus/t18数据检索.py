from typing import Tuple, List, Dict
from FlagEmbedding import BGEM3FlagModel
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker


# 客户端
def get_milvus_client():
    client = MilvusClient(
        uri="http://192.168.10.129:19530/",
        token=""
    )
    return client


# bgem3模型
def get_bge_m3_model():
    return BGEM3FlagModel(model_name_or_path=r"D:/modelhub/models/BAAI--bge-m3",
                          return_dense=True,
                          return_sparse=True)


# 参数向量化
def encode_query(model, query: str) -> Tuple[List[float], Dict[int, float]]:
    all_vectors = model.encode([query], return_dense=True, return_sparse=True)
    # 仅改这一行：强制转为float32，和集合字段精度对齐
    dense = __import__('numpy').array(all_vectors["dense_vecs"][0], dtype='float32')
    sparse = all_vectors["lexical_weights"][0]
    return dense, sparse


# 稠密检索
def dense_vector_search_example(client, query: str, limit: int = 5):
    dense, _ = encode_query(get_bge_m3_model(), query)
    result = client.search(
        collection_name="demo_collection", data=[dense], anns_field="vector", limit=limit,
        search_params={"metric_type": "L2"}, output_fields=["id", "text", "metadata"]
    )
    return result[0]


# 稀疏检索
def sparse_vector_search_example(client, query: str, limit: int = 5):
    _, sparse = encode_query(get_bge_m3_model(), query)
    result = client.search(
        collection_name="demo_collection", data=[sparse], anns_field="sparse_vector", limit=limit,
        search_params={"metric_type": "IP"}, output_fields=["id", "text", "metadata"]
    )
    return result[0]


# 混合检索
def hybrid_vector_search_example_rrf(client, query: str, limit: int = 5):
    dense, sparse = encode_query(get_bge_m3_model(), query)
    dense_request = AnnSearchRequest(data=[dense], anns_field="vector", param={"metric_type": "L2"}, limit=limit)
    sparse_request = AnnSearchRequest(data=[sparse], anns_field="sparse_vector", param={"metric_type": "IP"},
                                      limit=limit)
    result = client.hybrid_search(
        collection_name="demo_collection", reqs=[dense_request, sparse_request],
        ranker=RRFRanker(k=60), limit=limit, output_fields=["id", "text", "metadata"]
    )
    return result[0]


# 打印结果
def print_hits(title: str, hits: List[dict]):
    print("\n" + "=" * 20)
    print(title)
    print("=" * 20)
    for i, hit in enumerate(hits, start=1):
        entity = hit.get("entity", {})
        print({
            "rank": i,
            "id": entity.get("id"),
            "distance": hit.get("distance"),
            "text": entity.get("text"),
            "metadata": entity.get("metadata"),
        })


if __name__ == "__main__":
    client = get_milvus_client()
    # model = get_bge_m3_model()
    # dense, sparse = encode_query(model, "你好，世界")
    # print(dense)
    # print(sparse)
    # result = sparse_vector_search_example(client, "监护关系")
    # print_hits("稀疏向量", result)
    result = hybrid_vector_search_example_rrf(client, "监护关系")
    print_hits("混合检索", result)
