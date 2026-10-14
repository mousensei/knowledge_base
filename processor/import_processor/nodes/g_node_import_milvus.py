from processor.import_processor.base import BaseNode
from processor.import_processor.state import ImportGraphState


class NodeImportMilvus(BaseNode):
    """
    导入向量库节点：数据持久化
    """

    name = "node_import_milvus"

    def process(self, state: ImportGraphState):
        print("node_import_milvus...导入milvus节点...")

        return state
