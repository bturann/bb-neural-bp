import pytest


@pytest.fixture(autouse=True)
def local_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("BBNBP_CACHE_DIR", str(tmp_path / "cache"))
