from pathlib import Path
from typing import Optional

from prefect.filesystems import RemoteFileSystem
from prefect.logging import get_run_logger

REMOTE_STORAGE_BLOCK_NAME = "pipeline-remote-storage"


def upload_to_remote_storage(
    local_path: Path,
    remote_folder: str,
    block_name: str = REMOTE_STORAGE_BLOCK_NAME,
) -> Optional[str]:
    """
    Uploads ``local_path`` to a RemoteFileSystem block if available.

    Returns the remote URI (basepath + relative path) when the upload succeeds,
    otherwise ``None`` when the block is not configured.
    """
    logger = get_run_logger()
    try:
        remote_fs = RemoteFileSystem.load(block_name)
    except ValueError:
        logger.info(
            "Remote storage block '%s' non configurato: salto upload di %s",
            block_name,
            local_path,
        )
        return None

    relative_path = f"{remote_folder.strip('/')}/{local_path.name}"
    remote_fs.write_path(relative_path, local_path.read_bytes())

    basepath = remote_fs.basepath.rstrip("/")
    remote_uri = f"{basepath}/{relative_path}"
    logger.info("File %s caricato su %s", local_path, remote_uri)
    return remote_uri
