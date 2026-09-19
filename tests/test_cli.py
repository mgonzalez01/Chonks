"""Tests for the chonks CLI dispatcher."""
import importlib

import pytest

import chonks.cli as cli
import chonks.ops.cli


def test_no_arguments_prints_usage_and_exits_1(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main([])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.splitlines()[0] == "usage: chonks {init,index,serve,doctor,report} [args...]"


def test_help_prints_usage_and_exits_0(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines()[0] == "usage: chonks {init,index,serve,doctor,report} [args...]"


def test_unknown_command_exits_2(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["bogus"])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "chonks: unknown command 'bogus'" in captured.err


def test_every_subcommand_module_has_main():
    assert set(chonks.ops.cli._MODULES) == {"init", "index", "serve", "doctor", "report"}
    for module_name in chonks.ops.cli._MODULES.values():
        module = importlib.import_module(module_name)
        assert callable(module.main)


def test_init_return_code_becomes_exit_code(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["init", "--yes"])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--yes requires --codebase" in captured.out
