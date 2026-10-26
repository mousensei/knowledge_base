import asyncio

import uvicorn
from fastapi import FastAPI
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


@app.get("/stream/{session_id}")
async def stream_by_session(session_id: str):
    async def event_generator():
        for i in range(10):
            yield f"data:这是你的第{i}条数据\n\n"
            await asyncio.sleep(1)
        yield f"data:[END]\n\n"

    async def error_event_generator():
        yield f"data:会话无效\n\n"
        await asyncio.sleep(1)
        yield f"data:[END]\n\n"

    if session_id == "123":
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        return StreamingResponse(error_event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
