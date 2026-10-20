import asyncio

from agents.mcp import MCPServerStreamableHttp

from config.bailian_mcp_config import mcp_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger


class NodeWebSearchMcp(NodeBase):
    """
    节点功能，调用外部搜索引擎补充信息
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # TODO
        logger.info(f"【{self.name}】节点逻辑")

        # 1 参数
        query = state.get("rewritten_query")
        docs = []  # 解析结果

        # 2 调用mcp服务
        response = asyncio.run(self._mcp_call(query))
        print(response)

        # 3 结果解析

        return {"web_search_docs": docs}

    async def _mcp_call(self, query):
        # mcp服务工具
        search_mcp_tool = MCPServerStreamableHttp(
            name="search_mcp",
            params={
                "url": mcp_config.mcp_base_url,
                "headers": {
                    "Authorization": f"Bearer {mcp_config.api_key}"
                },
                "timeout": 10
            },
            cache_tools_list=True,
            max_retry_attempts=3
        )

        # 链接服务
        await search_mcp_tool.connect()

        # 发送请求
        result = await search_mcp_tool.call_tool(
            tool_name="bailian_web_search",
            arguments={"query": query, "count": 3},
        )
        await search_mcp_tool.cleanup()
        return result


if __name__ == "__main__":
    nodeWebSearch = NodeWebSearchMcp()
    init_state = {
        "rewritten_query": "尚硅谷的通关攻略是什么"
    }
    response = nodeWebSearch.process(init_state)
