from pathlib import Path
from modelscope.hub.snapshot_download import snapshot_download

# Reuse the migrated snapshot offline; download only when no snapshot exists.
cached_snapshot = Path('D:/modelhub/models/BAAI--bge-m3/snapshots/master')
model_dir = snapshot_download(
    'BAAI/bge-m3',
    cache_dir='D:/modelhub',
    revision='master',
    local_files_only=cached_snapshot.is_dir(),
)
print(f"模型已下载到: {model_dir}")
