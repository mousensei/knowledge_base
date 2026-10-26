import asyncio
import threading
import time

import uvicorn
from fastapi import FastAPI, BackgroundTasks

app = FastAPI()


# 同步方法
def write_email(email, content):
    while True:
        print(f"正在给{email}发送邮件：{content}...")
        time.sleep(1)


@app.get("/send_email/{email}")
async def send_task(email: str, background_tasks: BackgroundTasks):
    print("发送邮件是一个漫长的任务，我会启动一个后台任务执行....")
    background_tasks.add_task(write_email, email, "hello my friend...")
    print(f"接口调用者线程id：{threading.current_thread().ident}")
    # time.sleep(5)
    await asyncio.sleep(5)
    print("同时返回结果给调用者....")
    return {"message": "任务已经启动，请耐等待结果"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
