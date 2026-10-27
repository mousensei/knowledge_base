from datetime import datetime
import os
import shutil
import uuid
from pathlib import Path
from typing import List, Dict, Any

import dotenv
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

from config.minio_config import minio_config
from processor.import_processor.main_graph import KBImportWorkflow
from utils.minio_utils import get_minio_client
from utils.task_utils import add_running_task, add_done_task, update_task_status, get_task_status, get_done_task_list, \
    get_running_task_list

dotenv.load_dotenv()

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
async def upload_file(background_task: BackgroundTasks, files: List[UploadFile] = File(...)):
    # 接收用户上传的文件
    data_base_root_dir = os.getenv("DATA_BASE_ROOT_DIR")
    data_dir = os.path.join(data_base_root_dir, datetime.now().strftime("%Y%m%d"))
    task_ids = []
    for file in files:
        task_id = str(uuid.uuid4())  # 任务id随机字符串
        task_ids.append(task_id)

        # 1 记录任务日志容器
        add_running_task(task_id, "upload_file")

        # 2 存放用户上传文件
        file_dir = os.path.join(data_dir, task_id)
        os.makedirs(file_dir, exist_ok=True)
        import_file_path = os.path.join(file_dir, file.filename)

        # 3 接收用户上传的文件到服务器
        with open(import_file_path, "wb") as file_buffer:
            shutil.copyfileobj(file.file, file_buffer)

        # 4 minio服务器备份
        try:
            minio_object_name = f"pdf_files/{datetime.now().strftime('%Y%m%d')}/{file.filename}"
            minio_client = get_minio_client()
            minio_bucket_name = minio_config.bucket_name
            minio_client.fput_object(
                bucket_name=minio_bucket_name,
                file_path=import_file_path,
                object_name=minio_object_name,
                content_type=file.content_type
            )
        except Exception as e:
            raise e
        # 5 记录任务日志容器
        add_done_task(task_id, "upload_file")

        # 6 调用导入工作流
        background_task.add_task(run_import_graph, task_id, file_dir, import_file_path)

    return {"code": 200, "message": f"文件上传成功", "task_ids": task_ids}


# 运行工作流
def run_import_graph(task_id: str, file_dir: str, import_file_path: str):
    print("import服务，运行import导入工作流...")
    try:
        init_state = {
            "task_id": task_id,
            "file_dir": file_dir,
            "import_file_path": import_file_path,
        }
        update_task_status(task_id, "processing")  # 前端轮询得到的任务状态
        import_workflow = KBImportWorkflow()
        # 日志输出节点运行状态
        for event in import_workflow.run(init_state, stream=True):
            for node_name, node_result in event.items():
                add_running_task(task_id, node_name)
        update_task_status(task_id, "completed")
    except Exception as e:
        update_task_status(task_id, "failed")


@app.get("/status/{task_id}")
async def get_task_progress(task_id: str):
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id)
    }
    return task_status_info


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
