from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import RedirectResponse, FileResponse

# api对象
webApp = FastAPI()


# 路径参数
@webApp.get("/item/{item_id}")  # rest
async def get_item(item_id: int, a: str):
    print(f"item接口参数:item:{item_id},a:{a}")
    return {"item_id": item_id, "a": a}


# 对象参数
class Item(BaseModel):
    name: str
    price: int
    is_offer: bool = False


@webApp.post("/items")  # rest
async def get_item(item: Item):
    print(f"items接口参数:item:{item}")
    return item


# 重定向
@webApp.get("/old-path")
async def old_path():
    print("old-path...")
    return RedirectResponse(url="/new-path", status_code=307)


@webApp.get("/new-path")
async def new_path():
    print("new-path...")
    # item = Item()
    # item.name = "zhang3"
    # item.price = 300
    return {"message": "success"}


# 文件输出
@webApp.get("/down/excel")
def excel_down():
    print("excel_down...")
    # path = "E:/demo.xlsx"
    return FileResponse(
        path="static/demo.xlsx",
        filename="demos.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


if __name__ == '__main__':
    import uvicorn  # python的nginx

    uvicorn.run(webApp, host="0.0.0.0", port=8000)
