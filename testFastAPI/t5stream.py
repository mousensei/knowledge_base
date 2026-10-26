import asyncio

from fastapi import FastAPI
from starlette.responses import StreamingResponse

# api对象
webApp = FastAPI()


# yield
async def hello_world():
    words = ["你", "好", ",", "这", "是", "流", "式", "输", "出"]
    for word in words:
        await asyncio.sleep(0.5)
        yield word.encode("utf-8")


@webApp.get("/stream")
async def stream():
    print("stream流式输出...")
    return StreamingResponse(hello_world(), media_type="text/event-stream")


if __name__ == '__main__':
    import uvicorn  # python的nginx

    uvicorn.run(webApp, host="0.0.0.0", port=8000)
