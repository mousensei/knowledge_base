import time

"""同步"""


def download_file(name):
    print(f"开始下载任务{name}...")
    time.sleep(3)
    print(f"下载任务完成{name}...")


time_start = time.time()

# 下载
download_file("文件1")
download_file("文件2")
download_file("文件3")

time_end = time.time()

print(f"下载总耗时:{time_end - time_start}")
