import threading

from fastapi import FastAPI

# api对象
webApp = FastAPI()


@webApp.get("/simple-async")
async def simple_async():
    print(f"异步:{threading.current_thread().ident}...")
    return {"message": "hello async"}


@webApp.get("/simple-sync")
def simple_sync():
    print(f"同步:{threading.current_thread().ident}...")
    return {"message": "hello sync"}


if __name__ == '__main__':
    import uvicorn  # python的nginx

    uvicorn.run(webApp, host="0.0.0.0", port=8000)
