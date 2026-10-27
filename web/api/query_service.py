from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from fastapi import Request
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, StreamingResponse

from processor.query_processor.main_graph import KBQueryWorkflow
from utils.mongo_history_utils import get_recent_messages
from utils.sse_utils import create_sse_queue, sse_generator
from utils.task_utils import update_task_status, TASK_STATUS_PROCESSING, get_task_result

# app
app = FastAPI(
    title="检索 API",
    description="此文档是掌柜智库检索流程的API接口说明"
)

# 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 前端参数对象
class QueryRequest(BaseModel):
    query: str
    session_id: str
    is_stream: bool


@app.get("/chat.html")
async def get_import_page():
    web_dir = Path(__file__).absolute().parent.parent
    import_html_path = web_dir / "page" / "chat.html"
    if not import_html_path.exists():
        raise HTTPException(status_code=404, detail="不好意思，您访问的页面不存在")
    return FileResponse(import_html_path)


@app.post("/query")
async def query(background_task: BackgroundTasks, query: QueryRequest):
    # 解析参数
    user_query = query.query
    session_id = query.session_id
    is_stream = query.is_stream

    # 调用工作流
    if is_stream:
        create_sse_queue(session_id)  # 注册工作流的sse输出队列

    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)  # 前端日志输出

    if is_stream:
        background_task.add_task(run_query_graph, session_id, user_query, is_stream)
        # 返回结果
        return {"message": "任务已开始，请耐性等待", "session_id": session_id}
    else:
        # 几乎不会采用这种模式
        run_query_graph(session_id, user_query, is_stream)
        answer = get_task_result(session_id, "answer", "")
        # 返回结果
        return {"message": "处理完成", "session_id": session_id, "is_stream": is_stream, "answer": answer}


# 调用检索工作流
def run_query_graph(session_id: str, user_query: str, is_stream: bool):
    print("侯台任务：调用检索工作流...")
    init_state = {
        "session_id": session_id,
        "original_query": user_query,
        "is_stream": is_stream
    }
    query_workflow = KBQueryWorkflow()
    for event in query_workflow.run(init_state, stream=is_stream):
        print("节点运行结果和状态...")
        # 可以将节点执行结果放入页面要求的done_list和running_list


@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/history/{session_id}")
async def history(session_id: str, request: Request):
    # 根据session_id查询mongodb，结果封装items
    records = get_recent_messages(session_id, limit=3)
    items = []
    for r in records:
        items.append({
            "_id": str(r.get("_id")) if r.get("_id") is not None else "",
            "session_id": r.get("session_id", ""),
            "role": r.get("role", ""),
            "text": r.get("text", ""),
            "rewritten_query": r.get("rewritten_query", ""),
            "item_names": r.get("item_names", []),
            "ts": r.get("ts")
        })
    return {"session_id": session_id, "items": items}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
