from typing import Literal

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from processor.import_processor.nodes.a_node_entry import NodeEntry
from processor.import_processor.nodes.b_node_pdf_to_md import NodePDFToMD
from processor.import_processor.nodes.c_node_md_img import NodeMDImg
from processor.import_processor.nodes.d_node_document_split import NodeDocumentSplit
from processor.import_processor.nodes.e_node_item_name_recognition import NodeItemNameRecognition
from processor.import_processor.nodes.f_node_bge_embedding import NodeBGEEmbedding
from processor.import_processor.nodes.g_node_import_milvus import NodeImportMilvus
from processor.import_processor.state import ImportGraphState


class KBImportWorkflow:
    def __init__(self, config=None):
        self._compiled_graph = None

    def run(self, state: ImportGraphState, stream: bool = False):
        """运行图"""
        print(f"运行图...是否流式输出:{stream}...")
        if stream:
            return self.graph.stream(state, stream_mode="updates")
        else:
            return self.graph.invoke(state)

    @property
    def graph(self):
        """  """
        print("计算属性graph...")
        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph()

        return self._compiled_graph

    @staticmethod
    def route_after_entry(state: ImportGraphState):  # ->Literal["b_node_pdf_to_md","c_node_md_img",END]
        print("入口节点的路由...")
        """路由：去b或者c或者END"""
        if state.get("is_md_read_enabled"):
            return "c_node_md_img"
        elif state.get("is_pdf_read_enabled"):
            return "b_node_pdf_to_md"
        else:
            return END

    def build_graph(self):
        print("创建图...")

        # 注册节点
        graph = StateGraph(ImportGraphState)
        graph.add_node("a_node_entry", NodeEntry())
        graph.add_node("b_node_pdf_to_md", NodePDFToMD())
        graph.add_node("c_node_md_img", NodeMDImg())
        graph.add_node("d_node_document_split", NodeDocumentSplit())
        graph.add_node("e_node_item_name_recognition", NodeItemNameRecognition())
        graph.add_node("f_node_bge_embedding", NodeBGEEmbedding())
        graph.add_node("g_node_import_milvus", NodeImportMilvus())

        # 设置边
        # graph.add_edge(START,"a_node_entry")
        graph.set_entry_point("a_node_entry")
        graph.add_conditional_edges("a_node_entry",
                                    self.route_after_entry,
                                    {"b_node_pdf_to_md": "b_node_pdf_to_md",
                                     "c_node_md_img": "c_node_md_img",
                                     END: END})
        graph.add_edge("b_node_pdf_to_md", "c_node_md_img")
        graph.add_edge("c_node_md_img", "d_node_document_split")
        graph.add_edge("d_node_document_split", "e_node_item_name_recognition")
        graph.add_edge("e_node_item_name_recognition", "f_node_bge_embedding")
        graph.add_edge("f_node_bge_embedding", "g_node_import_milvus")

        graph_compile = graph.compile()
        return graph_compile


if __name__ == "__main__":
    import_state = {"import_file_path": "D:\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册.pdf"}
    import_graph = KBImportWorkflow()

    # 流式输出
    response = import_graph.run(import_state, stream=True)
    for item in response:
        print(f'结果是：{item.get("file_title")}导入完成')

    # 非流式输出
    # response = import_graph.run(import_state, stream=False)
    # print(response)
