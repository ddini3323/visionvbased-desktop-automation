from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FileManager:
    def __init__(self, project_dir_name: str = "tjm-project") -> None:
        self.project_dir_name = project_dir_name
        self._project_dir: Optional[Path] = None

    def get_desktop_path(self) -> Path:
        """Return the real Desktop path, handling OneDrive folder redirection."""
        # Method 1: .NET Environment.GetFolderPath - respects OneDrive redirection
        try:
            import ctypes, ctypes.wintypes
            dll = ctypes.windll.shell32
            buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
            # CSIDL_DESKTOPDIRECTORY = 0x10 (actual Desktop, not virtual)
            dll.SHGetFolderPathW(0, 0x10, 0, 0, buf)
            p = Path(buf.value)
            if p.exists():
                return p
        except Exception:
            pass
        # Method 2: USERPROFILE + OneDrive subfolder
        for env_key in ("OneDrive", "ONEDRIVE", "OneDriveConsumer", "OneDriveCommercial"):
            val = os.environ.get(env_key, "")
            if val:
                p = Path(val) / "Desktop"
                if p.exists():
                    return p
        # Method 3: USERPROFILE plain Desktop
        userprofile = os.environ.get("USERPROFILE", "")
        if userprofile:
            p = Path(userprofile) / "Desktop"
            if p.exists():
                return p
        return Path.home() / "Desktop"

    def setup_project_dir(self) -> Path:
        project_dir = self.get_desktop_path() / self.project_dir_name
        project_dir.mkdir(parents=True, exist_ok=True)
        self._project_dir = project_dir
        logger.info("Project directory: %s", project_dir)
        return project_dir

    @property
    def project_dir(self) -> Path:
        if self._project_dir is None:
            return self.setup_project_dir()
        return self._project_dir

    def get_save_path(self, post) -> Path:
        return self.project_dir / f"post_{post.id}.txt"

    def handle_existing(self, path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        counter = 1
        while True:
            candidate = parent / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    def format_content(self, post) -> str:
        return f"Title: {post.title}\n\n{post.body}"
