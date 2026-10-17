from gradio import Workflow
from langgraph.constants import END
from langgraph.graph import StateGraph

from processor.import_processor.main_graph import KBImportWorkflow
from processor.query_processor.nodes.a_node_item_name_confirm import NodeItemNameConfirm
from processor.query_processor.nodes.b_node_search_embedding import NodeSearchEmbedding
from processor.query_processor.nodes.c_node_search_embedding_hyde import NodeSearchEmbeddingHyde
from processor.query_processor.nodes.d_node_web_search_mcp import NodeWebSearchMcp
from processor.query_processor.nodes.e_node_rrf import NodeRrf
from processor.query_processor.nodes.f_node_rerank import NodeRerank
from processor.query_processor.nodes.g_node_answer_output import NodeAnswerOutput
from processor.query_processor.state import QueryGraphState


class KBQueryWorkflow:

    def __init__(self):
        print("KBQueryWorkflow，init...")
        # 声明图
        self.workflow = StateGraph(QueryGraphState)

        # 声明节点
        self._init_nodes()

        # 注册节点
        self._regist_nodes()

        # 注册边
        self._setup_routes()

        # 编译图
        self._compile_app = None

    def run(self, inital_state: QueryGraphState, stream: bool = False) -> QueryGraphState:
        if not self._compile_app:
            self._compile()
        if stream:
            return self._compile_app.stream(inital_state)
        else:
            return self._compile_app.invoke(inital_state)

    def _init_nodes(self):
        print("查询模块,初始化节点...")
        self.node_item_name_confirm = NodeItemNameConfirm()
        self.node_search_embedding = NodeSearchEmbedding()
        self.node_search_embedding_hyde = NodeSearchEmbeddingHyde()
        self.node_web_search_mcp = NodeWebSearchMcp()
        self.node_rrf = NodeRrf()
        self.node_rerank = NodeRerank()
        self.node_answer_output = NodeAnswerOutput()

    def _regist_nodes(self):
        self.workflow.add_node("node_item_name_confirm", self.node_item_name_confirm)
        self.workflow.add_node("node_search_embedding", self.node_search_embedding)
        self.workflow.add_node("node_search_embedding_hyde", self.node_search_embedding_hyde)
        self.workflow.add_node("node_web_search_mcp", self.node_web_search_mcp)
        self.workflow.add_node("node_rrf", self.node_rrf)
        self.workflow.add_node("node_rerank", self.node_rerank)
        self.workflow.add_node("node_answer_output", self.node_answer_output)

    def _setup_routes(self):
        self.workflow.set_entry_point("node_item_name_confirm")
        # 条件边
        self.workflow.add_conditional_edges("node_item_name_confirm",
                                            self._route_after_name_confirm,
                                            {
                                                "node_search_embedding": "node_search_embedding",
                                                "node_search_embedding_hyde": "node_search_embedding_hyde",
                                                "node_web_search_mcp": "node_web_search_mcp",
                                                "node_answer_output": "node_answer_output",
                                            }
                                            )
        self.workflow.add_edge("node_search_embedding", "node_rrf")
        self.workflow.add_edge("node_search_embedding_hyde", "node_rrf")
        self.workflow.add_edge("node_web_search_mcp", "node_rrf")
        self.workflow.add_edge("node_rrf", "node_rerank")
        self.workflow.add_edge("node_rerank", "node_answer_output")
        self.workflow.add_edge("node_answer_output", END)

    def _route_after_name_confirm(self, state: QueryGraphState):
        print("查询工作流条件路由，_route_after_name_confirm...")
        if state.get("answer"):
            return "node_answer_output"
        else:
            return ["node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"]

    def _compile(self):
        print("编译查询工作流图...")
        if not self._compile_app:
            self._compile_app = self.workflow.compile()
        return self._compile_app


if __name__ == "__main__":
    # 调用图
    workflow = KBQueryWorkflow()
    response = workflow.run({"original_query": "哥们儿，华为擎云B530计算机这个东东咋鼓捣上啊？", "session_id": "123"},
                            stream=True)
    for res in response:
        print(f"流式输出:{res}")

    # 画图
    # print(workflow.compile().get_graph().draw_ascii())
