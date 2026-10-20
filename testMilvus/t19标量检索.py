from pymilvus import MilvusClient


def get_milvus_client():
    client = MilvusClient(
        uri="http://192.168.222.99:19530", token=""
    )
    return client


def scalar_query_examples(client, like_keyword: str = "大模型"):
    # like_res = client.query(
    #     collection_name="demo_collection",
    #     filter=f'text like "%{like_keyword}%"',
    #     output_fields=["id", "text"]
    # )
    # for row in like_res:
    #     print({"id": row.get("id"), "text": row.get("text")})

    like_res = client.query(
        collection_name="demo_collection",
        filter=f'metadata["source"] like "%sample%"',
        output_fields=["id", "text"]
    )
    for row in like_res:
        print({"id": row.get("id"), "text": row.get("text")})


if __name__ == "__main__":
    client = scalar_query_examples(get_milvus_client(), "监护关系")
