import asyncio
import time

"""异步"""


async def download_file(name):
    print(f"开始下载任务{name}...")
    await asyncio.sleep(3)
    print(f"下载任务完成{name}...")


# 异步并行函数
async def main():
    await asyncio.gather(
        download_file("文件1"),
        download_file("文件2"),
        download_file("文件3")
    )


time_start = time.time()
# 启动下载
asyncio.run(main())
time_end = time.time()
print(f"下载总耗时:{time_end - time_start}")
