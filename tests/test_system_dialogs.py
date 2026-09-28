from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.auth import CurrentUser, require_edition
from api.routers import system_dialogs


def test_pick_folder_returns_absolute_selected_directory(monkeypatch, tmp_path):
    selected = tmp_path / "exports"
    selected.mkdir()
    monkeypatch.setattr(system_dialogs.platform, "system", lambda: "Darwin")
    run = Mock(return_value=Mock(returncode=0, stdout=f"{selected}\n"))
    monkeypatch.setattr(system_dialogs.subprocess, "run", run)

    assert system_dialogs._pick_folder(None) == str(selected.resolve())
    assert run.call_args.args[0][0] == "osascript"


def test_pick_folder_treats_cancel_as_no_change(monkeypatch):
    monkeypatch.setattr(system_dialogs.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        system_dialogs.subprocess,
        "run",
        Mock(return_value=Mock(returncode=0, stdout="")),
    )

    assert system_dialogs._pick_folder(None) is None


def test_pick_folder_rejects_non_directory_result(monkeypatch, tmp_path):
    missing = tmp_path / "missing"
    monkeypatch.setattr(system_dialogs.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        system_dialogs.subprocess,
        "run",
        Mock(return_value=Mock(returncode=0, stdout=str(missing))),
    )

    with pytest.raises(HTTPException) as exc:
        system_dialogs._pick_folder(None)
    assert exc.value.status_code == 400


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_loopback_hosts_are_accepted(host):
    assert system_dialogs._is_loopback(host)


def test_remote_host_is_rejected():
    assert not system_dialogs._is_loopback("192.168.1.5")


def test_folder_picker_route_returns_selected_absolute_path(app, monkeypatch, tmp_path):
    selected = tmp_path / "exports"
    selected.mkdir()
    app.dependency_overrides[require_edition] = lambda: CurrentUser(
        user_id="test", role="admin", edition_authenticated=True,
    )
    monkeypatch.setattr(system_dialogs, "_pick_folder", lambda initial: str(selected))

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        response = client.post("/api/system/folder-picker", json={"initial_dir": None})

    assert response.status_code == 200
    assert response.json() == {"path": str(selected)}


def test_folder_picker_route_refuses_remote_clients(app):
    app.dependency_overrides[require_edition] = lambda: CurrentUser(
        user_id="test", role="admin", edition_authenticated=True,
    )

    with TestClient(app, client=("192.168.1.5", 50000)) as client:
        response = client.post("/api/system/folder-picker", json={"initial_dir": None})

    assert response.status_code == 403
