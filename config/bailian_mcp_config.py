# from dataclasses import dataclass
# import os
# from dotenv import load_dotenv
#
# load_dotenv()
#
#
# # 定义mcp的服务配置
# @dataclass
# class McpConfig:
#     mcp_base_url: str
#     api_key : str
#
# mcp_config = McpConfig(
#     mcp_base_url=os.getenv("MCP_DASHSCOPE_BASE_URL"),
#     api_key=os.getenv("OPENAI_API_KEY")
# )

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

# 获取项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 加载项目根目录的 .env
load_dotenv(BASE_DIR / ".env")


@dataclass
class McpConfig:
    mcp_base_url: str
    api_key: str


# 阿里云百炼 MCP 专用配置
mcp_config = McpConfig(
    mcp_base_url=os.getenv(
        "MCP_DASHSCOPE_BASE_URL", ""
    ).strip(),

    api_key=os.getenv(
        "DASHSCOPE_API_KEY", ""
    ).strip()
)
