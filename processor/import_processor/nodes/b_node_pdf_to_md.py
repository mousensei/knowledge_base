import time
import zipfile
from pathlib import Path

import requests

from config.mineru_config import mineru_config
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, PdfConversionError
from processor.import_processor.state import ImportGraphState


class NodePDFToMD(BaseNode):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """

    name = "node_pdf_to_md"

    def process(self, state: ImportGraphState):
        print("node_pdf_to_md...pdf文档转md文档...")

        # 1 检查参数，pdf_path,file_dir
        pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state)

        # 2 上传mineru，轮询结果(下载地址)
        zip_url = self._step_2_upload_and_pull(pdf_path_obj)
        print(f"下载链接地址:{zip_url}")

        # 3 下载和解压
        md_path = self._step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)

        # 4 读取md文本内容
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # 5 更新state全局状态
        state["md_content"] = md_content
        state["md_path"] = md_path
        return state

    def _step_1_validate_paths(self, state: ImportGraphState):
        print("_step_1_validate_paths...")
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise StateFieldError(message="pdf_path路径参数为空")

        file_dir = state.get("file_dir")
        if not file_dir:
            raise StateFieldError(message="file_dir路径参数为空")

        pdf_path_obj = Path(pdf_path)
        file_dir_obj = Path(file_dir)

        if not pdf_path_obj.exists():
            raise FileProcessingError(message="pdf文件不存在")

        if not file_dir_obj.exists():
            raise FileProcessingError(message="输出路径不存在")

        return pdf_path_obj, file_dir_obj

    def _step_2_upload_and_pull(self, pdf_path_obj):
        print("_step_2_upload_and_pull...")
        api_token = mineru_config.api_token
        base_url = mineru_config.base_url
        request_upload_url = f"{base_url}/file-urls/batch"  # 申请批量上传链接
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_token}"
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm"
        }
        response = requests.post(request_upload_url, headers=header, json=data)
        if response.status_code != 200:  # http是否链接到服务器
            raise Exception(f"{request_upload_url}链接失败")
        result = response.json()
        if result.get("code") != 0:  # 是否申请成功
            raise Exception(f"被服务器拒绝")
        batch_id = result["data"]["batch_id"]  # 将来取下载地址的id
        file_urls = result["data"]["file_urls"]  # 上传链接地址(列表)

        # 上传文件
        with open(pdf_path_obj, "rb") as f:
            res_upload = requests.put(file_urls[0], data=f)
            if res_upload.status_code != 200:
                raise Exception(f"{file_urls[0]}链接失败")
            print("文件上传成功...")

        # 轮询获取任务结果(下载链接)
        pull_url = f"{base_url}/extract-results/batch/{batch_id}"
        start_time = time.time()  # 记录开始时间
        timeout_seconds = 600  # 最大超时时间
        pull_interval = 3  # 轮询间隔时间
        self.logger.info(f"【任务轮询】最大超时：{timeout_seconds}s，batch_id：{batch_id}")
        while True:
            cost_time = time.time() - start_time
            if cost_time > timeout_seconds:
                raise Exception("请求超时")
            try:
                res_pull = requests.get(pull_url, headers=header, timeout=10)
            except Exception as e:
                print("请求失败，重新尝试")
                time.sleep(pull_interval)
                continue
            if res_pull.status_code != 200:
                raise Exception("链接失败...")

            pull_data = res_pull.json()
            if pull_data.get("code") != 0:
                raise Exception("文档解析服务器异常...")
            extract_results = pull_data["data"]["extract_result"]
            extract_result = extract_results[0]
            extract_state = extract_result["state"]
            if extract_state == "done":
                print("任务完成，获得下载链接...")
                full_zip_url = extract_result["full_zip_url"]
                return full_zip_url
            elif extract_state == "failed":
                err_msg = extract_result.get("err_msg", "未知错误，无具体信息")
                raise PdfConversionError(f"【任务轮询】解析任务失败！batch_id：{batch_id}，错误信息：{err_msg}")
            else:
                self.logger.info(
                    f"【任务轮询】处理中... 已耗时{int(cost_time)}s，状态：{extract_state}， batch_id：{batch_id}")
                time.sleep(pull_interval)

    def _step_3_download_and_extract(self, zip_url, output_dir_obj, pdf_stem):
        print("_step_3_download_and_extract...")

        # 1 下载
        response = requests.get(zip_url)
        if response.status_code != 200:
            raise Exception("下载失败...")
        zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip"
        with open(zip_save_path, "wb") as f:
            f.write(response.content)

        # 2 创建输出目录
        extract_target_dir = output_dir_obj / f"{pdf_stem}"
        extract_target_dir.mkdir(parents=True, exist_ok=True)

        # 3 解压
        with zipfile.ZipFile(zip_save_path, "r") as zip_file_obj:
            zip_file_obj.extractall(extract_target_dir)
        print("解压完成...")

        # 4 重命名
        target_md_file = extract_target_dir / "full.md"
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
        target_md_file.replace(new_md_path)
        self.logger.info(f"【MD重命名】重命名成功，文件名：{pdf_stem}.md")

        return str(new_md_path.absolute())


if __name__ == "__main__":
    init_state = {
        "pdf_path": "D:\\hak180产品安全手册.pdf",
        "file_dir": r"D:\\output",
    }

    node = NodePDFToMD()
    # response = node.process(init_state)
    response = node(init_state)
    print(response)
