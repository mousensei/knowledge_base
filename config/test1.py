from dataclasses import dataclass, field

from config.minio_config import minio_config

endpint = minio_config.endpoint
print(endpint)
