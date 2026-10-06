import tomllib
from importlib.metadata import version
from pathlib import Path

from holos_tts import __main__ as entry
from holos_tts.constants import DISTRIBUTION_NAME, PROGRAM_NAME


def test_distribution_name_is_the_name_in_pyproject():
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    assert tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["name"] == DISTRIBUTION_NAME
    assert DISTRIBUTION_NAME != PROGRAM_NAME


def test_version_is_the_version_of_the_installed_package():
    assert entry._version() == version(DISTRIBUTION_NAME)
    assert entry._version() != "0"
