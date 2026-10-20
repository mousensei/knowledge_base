from pymilvus import MilvusClient, DataType


# 链接client
def get_milvus_client():
    client = MilvusClient(
        uri="http://192.168.222.99:19530/",
        token=""
    )
    return client


# 设置字段
def build_schema():
    schema = MilvusClient.create_schema(
        auto_id=True,
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1500)
    schema.add_field(field_name="metadata", datatype=DataType.JSON)
    return schema


# 设置索引
def build_index():
    index = MilvusClient.prepare_index_params()
    index.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="L2",  # 向量相似度函数
    )
    index.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",  # 向量相似度函数
    )
    return index


# 创建集合
def create_collection(client):
    client.drop_collection(collection_name="demo_collection")
    if not client.has_collection(collection_name="demo_collection"):
        client.create_collection(
            collection_name="demo_collection",
            schema=build_schema(),  # 表字段
            index_params=build_index()  # 表索引
        )
        print(client.list_collections())


# 测试
if __name__ == '__main__':
    client = get_milvus_client()
    # print(client.list_collections())  # 打印所有的集合
    create_collection(client)
