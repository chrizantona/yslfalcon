"""The local installer must preserve existing credentials and avoid public secret files."""

import importlib.util
import stat
from pathlib import Path

spec = importlib.util.spec_from_file_location("bootstrap", Path(__file__).parents[1] / "scripts/init_env.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def test_initialization_preserves_existing_credentials(tmp_path):
    assert bootstrap.initialize(tmp_path)
    before = (tmp_path / ".env").read_bytes()
    token = (tmp_path / ".local/access-token.txt").read_bytes()
    assert not bootstrap.initialize(tmp_path)
    assert (tmp_path / ".env").read_bytes() == before
    assert (tmp_path / ".local/access-token.txt").read_bytes() == token


def test_generated_secrets_are_private_and_origins_match_the_port(tmp_path):
    bootstrap.initialize(tmp_path, port=8123)
    assert stat.S_IMODE((tmp_path / ".env").stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / ".local/access-token.txt").stat().st_mode) == 0o600
    assert len((tmp_path / ".local/access-token.txt").read_text().strip()) >= 32
    assert "http://localhost:8123" in (tmp_path / ".env").read_text()
