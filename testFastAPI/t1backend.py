from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

# api对象
webApp = FastAPI()

# 跨域
webApp.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@webApp.get("/api/data")
def get_data():
    print("后端/api/data接口")
    return {"message": "后端数据/api/data接口", "status": "success"}


if __name__ == '__main__':
    import uvicorn  # python的nginx

    uvicorn.run(webApp, host="0.0.0.0", port=8001)
