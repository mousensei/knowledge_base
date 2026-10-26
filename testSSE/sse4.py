import asyncio
import threading

import uvicorn
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

app = FastAPI()
# 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

task_queue = {}


# 后台任务（AI生成工作流）
async def long_task(session_id):
    print("长耗时工作流,long_task开始...")
    queue = asyncio.Queue()
    task_queue[session_id] = queue

    # 生成答案
    for i in range(5):
        await asyncio.sleep(1)  # 模拟ai生成耗时
        msg = f"这是{session_id}工作流生成的第{i}条数据"
        await queue.put(msg)
    await queue.put(None)
    print("长耗时工作流,long_task结束...")


# 给前端的事件(输出结果)
async def event_generator(session_id: str):
    while session_id not in task_queue:
        await asyncio.sleep(1)
    # 从字典中找到session_id对应的答案队列queue
    queue = task_queue[session_id]
    while True:
        msg = await queue.get()
        if msg is None:
            break
        await asyncio.sleep(1)
        yield f"data:{msg}\n\n"


# 数据对象
class QueryRequest(BaseModel):
    question: str
    session_id: str


# 短链接
@app.post("/query")
async def query_by_question(query: QueryRequest, background_tasks: BackgroundTasks):
    question = query.question
    session_id = query.session_id
    print(f"短链接：接受用户问题{session_id}，启动后台工作流任务:{question}，立即返回结果...")
    background_tasks.add_task(long_task, session_id)
    return {"message": "我们已经接收到你的问题，请耐♥等待", "session_id": session_id}


# 长链接
@app.get("/stream/{session_id}")
async def stream_by_session(session_id: str):
    print(f"{threading.current_thread().ident}:我来问问，有结果了没？")
    return StreamingResponse(
        event_generator(session_id),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
