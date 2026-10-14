from pathlib import Path

from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, ValidationError
from processor.import_processor.state import ImportGraphState


class NodeEntry(BaseNode):
    """
    入口节点：任务分发
    """

    name = "node_entry"

    def process(self, state: ImportGraphState):
        print("node_entry...校验和判断传入参数...")
        import_file_path = state.get("import_file_path")

        if not import_file_path:
            raise StateFieldError(message=f"文件不存在：{import_file_path}")

        import_file_path_obj = Path(import_file_path)
        if not import_file_path_obj.exists():
            raise FileProcessingError(message="文件不存在")

        if import_file_path_obj.suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        elif import_file_path_obj.suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
        else:
            raise ValidationError(message=f"该文件的后缀格式{import_file_path_obj.suffix}不支持")

        state["file_title"] = import_file_path_obj.name
        state["file_dir"] = r"D:\\output"  # 到出目录：测试节点产出物
        return state
