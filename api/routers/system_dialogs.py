"""Local desktop dialogs used by the browser UI.

Browsers deliberately do not reveal an absolute path from their directory
picker.  Facet can return one because it runs on the same machine as the photo
library, but only a loopback request may make the server open a desktop dialog.
"""

import ipaddress
import os
import platform
import shutil
import subprocess
import threading
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from api.auth import CurrentUser, require_edition

router = APIRouter(tags=["system-dialogs"])

_dialog_lock = threading.Lock()


class FolderPickerRequest(BaseModel):
    initial_dir: Optional[str] = None


class FolderPickerResponse(BaseModel):
    path: Optional[str]


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.lower() == "localhost"


def _pick_folder(initial_dir: Optional[str]) -> Optional[str]:
    initial = initial_dir if initial_dir and os.path.isdir(initial_dir) else os.path.expanduser("~")
    system = platform.system()

    if system == "Darwin":
        command = [
            "osascript",
            "-e",
            'on run argv\ntry\nset picked to choose folder with prompt "Choose target folder" default location POSIX file (item 1 of argv)\nreturn POSIX path of picked\non error number -128\nreturn ""\nend try\nend run',
            initial,
        ]
    elif system == "Windows":
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$d=New-Object System.Windows.Forms.FolderBrowserDialog; "
            "$d.SelectedPath=$args[0]; "
            "if($d.ShowDialog() -eq 'OK'){[Console]::Write($d.SelectedPath)}"
        )
        command = ["powershell", "-NoProfile", "-Command", script, initial]
    else:
        zenity = shutil.which("zenity")
        if not zenity:
            raise HTTPException(
                status_code=501,
                detail="A system folder picker is unavailable on this host (zenity is not installed)",
            )
        command = [zenity, "--file-selection", "--directory", "--filename", initial + os.sep]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise HTTPException(status_code=503, detail="Could not open the system folder picker") from exc

    # Cancelling is a normal empty result. zenity uses exit 1; the AppleScript
    # catches macOS's -128 cancellation and also returns an empty string.
    if not result.stdout.strip() and result.returncode in (0, 1):
        return None
    if result.returncode != 0:
        raise HTTPException(status_code=503, detail="The system folder picker failed to open")

    selected = os.path.abspath(os.path.expanduser(result.stdout.strip().rstrip("\r\n")))
    if not os.path.isdir(selected):
        raise HTTPException(status_code=400, detail="The selected path is not a directory")
    return selected


@router.post("/api/system/folder-picker", response_model=FolderPickerResponse)
def api_folder_picker(
    body: FolderPickerRequest,
    request: Request,
    user: CurrentUser = Depends(require_edition),
):
    """Open the server host's native folder chooser for a local operator."""
    host = request.client.host if request.client else ""
    if not _is_loopback(host):
        raise HTTPException(
            status_code=403,
            detail="The system folder picker is available only when Facet is opened locally",
        )
    if not _dialog_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A system folder picker is already open")
    try:
        return FolderPickerResponse(path=_pick_folder(body.initial_dir))
    finally:
        _dialog_lock.release()
