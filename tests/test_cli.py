from click.testing import CliRunner

from shenshang.cli import main


def test_binary():
  runner = CliRunner()
  result = runner.invoke(main, ['binary', '-h'])
  assert result.exit_code == 0
  # assert result.output == 'Hello Peter!\n'
