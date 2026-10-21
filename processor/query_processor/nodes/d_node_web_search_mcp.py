import asyncio
import json

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
        result = asyncio.run(self._mcp_call(query))
        # print(result)

        # 3 结果解析
        json_text_str = result.content[0].text
        result_json_obj = json.loads(json_text_str)
        pages = result_json_obj.get("pages")
        for page in pages:
            snippet = (page.get("snippet") or "").strip()
            url = (page.get("url") or "").strip()
            title = (page.get("title") or "").strip()
            docs.append({
                "url": url,
                "title": title,
                "snippet": snippet,
            })
        # print(docs)
        return {"web_search_docs": docs}

    # mcp工具调用
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
            tool_name="search_pro",
            arguments={"query": query},
        )
        await search_mcp_tool.cleanup()
        return result


if __name__ == "__main__":
    nodeWebSearchMcp = NodeWebSearchMcp()
    init_state = {
        "rewritten_query": "黑神话悟空通关攻略"
    }
    response = nodeWebSearchMcp(init_state)
