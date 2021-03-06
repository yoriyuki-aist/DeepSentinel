from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from deep_sentinel import utils


# pytest's temporary directory fixture (tmpdir) is a py.path.LocalPath object
# This is a workaround to use tempfile.TemporaryDirectory
@pytest.fixture
def tmp_dir() -> 'Path':
    td = TemporaryDirectory()
    yield Path(td.name)
    td.cleanup()


class TestToPath(object):

    @pytest.mark.parametrize(
        "path", ["a", "a/b", Path("a"), Path("a/b")]
    )
    def test_normal(self, path):
        actual = utils.to_path(path)
        assert isinstance(actual, Path)

    @pytest.mark.parametrize(
        "path, exc", [
            (1, TypeError),
            (1.0, TypeError),
            (list(), TypeError),
            (set(), TypeError),
            (dict(), TypeError),
        ]
    )
    def test_exc(self, path, exc):
        with pytest.raises(exc):
            utils.to_path(path)


class TestToAbsolute(object):

    @pytest.mark.parametrize(
        "path,expected", [
            (Path("a"), Path.cwd() / "a"),
            (Path("../../b"), Path.cwd().parent.parent / "b"),
            (Path("~/c"), Path.home() / "c"),
            (Path("./c"), Path.cwd() / "c"),
            (Path("/b"), Path("/b")),
        ]
    )
    def test_normal(self, path, expected):
        actual = utils.to_absolute(path)
        assert isinstance(actual, Path)
        assert actual.is_absolute()
        assert actual == expected

    @pytest.mark.parametrize(
        "path, exc", [
            (1, AttributeError),
            (1.0, AttributeError),
            (list(), AttributeError),
            (set(), AttributeError),
            (dict(), AttributeError),
        ]
    )
    def test_error(self, path, exc):
        with pytest.raises(exc):
            utils.to_absolute(path)


@pytest.mark.parametrize(
    "path,exists", [
        (Path("a"), True),
        (Path("b"), False),
        (Path("../"), True),
    ]
)
def test_exists(path, exists, tmp_dir):
    path = tmp_dir / path
    if exists and not path.exists():
        path.mkdir(parents=True)
    actual = utils.exists(path)
    assert actual is exists


class TestMkdir(object):

    @pytest.mark.parametrize(
        "path,exists", [
            (Path("a"), True),
            (Path("b"), False),
            (Path("c/d"), False),
            (Path("c/d"), True),
            (Path("d/../e/b"), False),
        ]
    )
    def test_normal(self, path, exists, tmp_dir):
        path = tmp_dir / path
        if exists:
            path.mkdir(parents=True)
        actual = utils.mkdir(path)
        assert isinstance(actual, Path)
        assert actual.exists()
        assert actual.is_absolute()
        assert str(actual) == str(path)

    @pytest.mark.parametrize(
        "path", ["a", "a/b", "c/../d"]
    )
    def test_error(self, path, tmp_dir):
        path = tmp_dir / Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as fp:
            fp.write("Hello")
        assert path.exists()
        assert path.is_file()
        with pytest.raises(NotADirectoryError):
            utils.mkdir(path)


class TestAvoidOverride(object):

    @pytest.mark.parametrize(
        "path", ["a", "./b", "c/../d"]
    )
    def test_no_dup(self, path, tmp_dir):
        path = tmp_dir / path
        actual = utils.avoid_override(path)
        assert isinstance(actual, Path)
        assert actual == path

    @pytest.mark.parametrize(
        "path", ["a", "./b", "c/../d"]
    )
    def test_dup(self, path, tmp_dir, capsys):
        path = tmp_dir / path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as fp:
            fp.write("hello")
        actual = utils.avoid_override(path)
        assert isinstance(actual, Path)
        assert path != actual
        assert path.parent == actual.parent
