from pathlib import Path

import uvicorn
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent
print(BASE_DIR)
static_dir = BASE_DIR / "static"
print(static_dir)

app = FastAPI()

# 挂载目录
app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
