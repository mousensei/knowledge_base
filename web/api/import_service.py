from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

# app
app = FastAPI(
    title="导入 API",
    description="此文档是掌柜智库导入流程的API接口说明"
)

# 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/import.html")
async def get_import_page():
    web_dir = Path(__file__).absolute().parent.parent
    import_html_path = web_dir / "page" / "import.html"
    if not import_html_path.exists():
        raise HTTPException(status_code=404, detail="不好意思，您访问的页面不存在")
    return FileResponse(import_html_path)


@app.post("/upload")
async def upload_file(backgroundTasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    # 调用工作流
    pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
