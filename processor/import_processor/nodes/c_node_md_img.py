import base64
import logging
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Tuple, Deque

from minio.deleteobjects import DeleteObject

from config.lm_config import lm_config
from config.minio_config import minio_config
from processor.import_processor.base import BaseNode
from processor.import_processor.state import ImportGraphState
from utils.llm_utils import get_llm_client
from utils.minio_utils import get_minio_client


class NodeMDImg(BaseNode):
    """
    MarkDown图片处理节点：多模态图片理解
    """

    name = "node_md_img"

    def process(self, state: ImportGraphState):
        print("node_md_img...图片处理节点...")

        # 1 参数处理
        md_content, md_path_obj, images_dir = self._step_1_get_content(state)

        # 2 扫描图片，所有md图片位置信息
        target_images = self._step_2_scan_images(md_content, images_dir)
        # print(target_images)

        # 3 模型理解,生成图片摘要
        summaries = self._step_3_generate_summaries(md_path_obj.stem, target_images)  # dict[str,str]
        # print(summaries)

        # 4 上传minio,上传图片,替换md内容
        new_md_content = self._step_4_upload_and_replace(md_path_obj.stem, target_images, summaries, md_content)
        # print("==========================替换结果=========================")
        # print(new_md_content)

        # 5 备份,更新state
        new_md_file_name = self._step_5_backup_new_md_file(state["md_path"], new_md_content)

        state["md_path"] = new_md_file_name
        state["md_content"] = new_md_content
        return state

    # 步骤1检查参数
    def _step_1_get_content(self, state: ImportGraphState) -> tuple[str, Any, Any]:
        print("图片节点，_step_1_get_content...")
        # 1 校验参数
        md_path = state["md_path"]
        if not md_path:
            raise Exception("md_path参数不存在或者错误...")
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise Exception("路径不存在...")

        md_content = state["md_content"]
        if not md_content:
            raise Exception("md_content参数不存在...")

        # 2 拼接images_dir
        images_dir = md_path_obj.parent / "images"

        return md_content, md_path_obj, images_dir

    # 步骤2扫描图片
    def _step_2_scan_images(self, md_content, images_dir) -> list[(str, str, tuple[str, str])]:
        print("图片节点，_step_2_scan_images...")
        target_images = []  # 返回结果列表

        # 1 扫描
        for image_file in os.listdir(images_dir):
            file_ext = os.path.splitext(image_file)[1].lower()
            # 判断图片格式是否支持
            if file_ext not in self.config.image_extensions:
                print(f"发现不支持的图片格式:{file_ext}")
                continue
            # 判断图片是否在md文档中被引用
            img_path = images_dir / image_file  # 当前图片的完整path路径
            context = self._find_image_in_md(md_content, image_file)  # 查找md文档中的图片是否存在,并且带回上下文
            if not context:
                print("图片没在md文档中被引用...")
                continue
            target_images.append((image_file, img_path, context))

        return target_images

    # 步骤2方法1查看md中是否有该图片
    def _find_image_in_md(self, md_content, image_file, context_len: int = 100) -> Tuple[str, str]:
        # 1 md文档中找到图片位置  ![MD图片处理流程](images/MD图片处理流程.png)
        partern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + ".*?\)")
        match = partern.search(md_content)

        # 2 获得图片上下文
        if not match:
            return None
        start, end = match.span()  # 图片开始位置和结束位置
        pre_text = md_content[max(0, (start - context_len)):start]
        post_text = md_content[end:min(len(md_content), (end + context_len))]
        return pre_text, post_text

    # 步骤3视觉vlm生成摘要
    def _step_3_generate_summaries(self, md_stem: str, target_images: list[(str, str, tuple[str, str])]) -> dict[
        str, str]:
        summaries = {}
        request_deque = deque()
        for image_file, image_path, context in target_images:
            # 滑动窗口限流(每分钟10次)
            self._apply_api_rate_limit(request_deque, max_requests=10)
            # 调用vlm模型
            summaries[image_file] = self._summerize_image(image_path, root_folder=md_stem, context=context)

        return summaries

    # 步骤3方法1滑动窗口
    def _apply_api_rate_limit(
            self,
            request_times: Deque[float],
            max_requests: int,
            window_seconds: int = 60
    ) -> None:
        current_time = time.time()

        # 1. 清理滑动窗口外的过期请求时间戳，保证队列仅存窗口内的请求
        while request_times and current_time - request_times[0] >= window_seconds:
            request_times.popleft()

        # 2. 窗口内请求数达上限，计算并阻塞等待剩余时间
        if len(request_times) >= max_requests:
            # 计算需要等待的时长（窗口总时长 - 最早请求已存在的时长）
            sleep_duration = window_seconds - (current_time - request_times[0])
            if sleep_duration > 0:
                logging.getLogger().info(
                    f"触发API速率限制，窗口{window_seconds}秒内最多{max_requests}次，需等待：{sleep_duration:.2f} 秒")
                time.sleep(sleep_duration)
                # 等待后更新当前时间，重新清理过期请求（避免等待期间有请求过期）
                current_time = time.time()
                while request_times and current_time - request_times[0] >= window_seconds:
                    request_times.popleft()

        # 3. 记录当前请求时间戳，加入滑动窗口队列
        request_times.append(current_time)
        logging.getLogger().info(f"API请求时间戳已记录，当前{window_seconds}秒窗口内请求数：{len(request_times)}")

    # 步骤3方法2视觉模型图片摘要
    def _summerize_image(self, image_path, root_folder, context) -> str:

        # 图片
        with open(image_path, "rb") as image_file:
            base64_str = base64.b64encode(image_file.read()).decode("utf-8")
        # 提示词
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""这是"{root_folder}"文件中的一张图片，图片上文部分为"{context[0]}"，下文部分为"{context[1]}"，请用中文简要总结这张图片的内容，用于 Markdown 图片标题。"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_str}"
                        }
                    }
                ]
            }
        ]
        # 模型(vlm)
        vlm = get_llm_client(lm_config.vl_model)

        # 调用结果
        response = vlm.invoke(messages)
        return response.content.strip().replace("\n", "")

    # 步骤4上传和替换图片信息
    def _step_4_upload_and_replace(self, doc_stem: str, target_images: list[(str, str, tuple[str, str])],
                                   summaries: dict[str, str],
                                   md_content: str):
        # 1 minio客户端
        minio_client = get_minio_client()
        minio_img_dir = minio_config.img_dir
        if minio_img_dir:
            upload_dir = f"{minio_img_dir}/{doc_stem}"
        else:
            upload_dir = doc_stem

        # 2 清理可能重复目录
        self._clean_minio_directory(minio_client, upload_dir)

        # 3 上传图片，返回urls
        urls = self._upload_images_batch(minio_client, upload_dir, target_images)
        # print("=========================上传结果urls=========================")
        # print(urls)

        # 4 将urls和summaries合并
        image_info = self._merge_summary_and_url(summaries, urls)
        # print("=========================合并结果=========================")
        # print(image_info)

        # 5 替换md中的图片信息
        new_md_content = self._process_md_file(md_content, image_info)

        return new_md_content

    # 步骤4方法1清理
    def _clean_minio_directory(self, minio_client, upload_dir):
        objects_to_delete = minio_client.list_objects(minio_config.bucket_name, prefix=upload_dir, recursive=True)
        delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]
        if delete_list:
            generator = minio_client.remove_objects(minio_config.bucket_name, delete_list)
            for obj in generator:
                print(f"{obj.object_name}已经被删除...")

    # 步骤4方法2上传
    def _upload_images_batch(self, minio_client, upload_dir: str, target_images: list[(str, str, tuple[str, str])]):
        urls = {}
        for image_file, image_path, _ in target_images:
            object_name = f"{upload_dir}/{image_file}"
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=object_name,
                file_path=image_path,
                content_type=f"image/{os.path.splitext(image_path)[1][1:]}")
            url = f"http://{minio_config.endpoint}/{minio_config.bucket_name}/{object_name}"
            urls[image_file] = url
        return urls

    # 步骤4方法3合并
    def _merge_summary_and_url(self, summaries: dict[str, str], urls: [str, str]) -> dict[str, tuple[str, str]]:
        print("图片节点，步骤4合并摘要和url...")
        image_info = {}
        for image_file, summary in summaries.items():
            if url := urls.get(image_file):
                image_info[image_file] = (summary, url)
        return image_info

    # 步骤4方法4替换
    def _process_md_file(self, md_content: str, image_info: dict[str, tuple[str, str]]) -> str:
        for image_file, (summary, url) in image_info.items():
            partern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + ".*?\)")
            md_content = partern.sub(lambda m: f"![{summary}]({url})", md_content)
        return md_content

    # 步骤5备份新md文件
    def _step_5_backup_new_md_file(self, md_path, new_md_content: str):
        print("图片节点，备份新文档")
        new_md_file_name = os.path.splitext(md_path)[0] + "_new.md"
        with open(new_md_file_name, "w", encoding="utf-8") as f:
            f.write(new_md_content)
        return new_md_file_name


if __name__ == "__main__":
    # md_path = "D:\output\hak180产品安全手册\hak180产品安全手册.md"
    # md_path_obj = Path(md_path)
    # print(type(md_path_obj.parent))
    # images_dir = md_path_obj.parent / "images"
    # print(type(images_dir))
    # print(images_dir)
    with open("D:\output\hak180产品安全手册\hak180产品安全手册.md", "r", encoding="utf-8") as f:
        md_content = f.read()
    init_state = {
        "md_path": "D:\output\hak180产品安全手册\hak180产品安全手册.md",
        "md_content": md_content
    }

    node = NodeMDImg()
    # response = node.process(init_state)
    response = node(init_state)
    # print(response)
