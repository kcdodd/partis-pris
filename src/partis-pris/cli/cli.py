
try:
  from importlib.metadata import metadata, requires, PackageNotFoundError

except ImportError:
  from importlib_metadata import metadata, requires, PackageNotFoundError

import click

from partis.utils import (
  HINT_LEVELS_DESC,
  init_logging,
  log_levels,
  getLogger,
  LogListHandler,
  ModelHint,
  VirtualEnv,
  MutexFile )

from ..env_spec import EnvSpec

from .add import add

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@click.group()
#...............................................................................
@click.option(
  '-v', '--verbosity',
  help = f"log verbosity",
  type = click.Choice(
    [ k.lower() for k in HINT_LEVELS_DESC.keys() ],
    case_sensitive = False ),
  default = 'info',
  show_default = True )
#...............................................................................
@click.option(
  '--color',
  'with_color',
  help = """Enable color log output.
    If not set, an attempt is made to detect whether or not to generate terminal
    color codes.""",
  is_flag = True,
  flag_value = True,
  default = None)
@click.option(
  '--no-color',
  'with_color',
  help = "Disable color log output",
  is_flag = True,
  flag_value = False,
  default = None)
#...............................................................................
@click.option(
  '--ascii',
  help = "Disable non-ascii log output where possible.",
  type = bool,
  default = False,
  is_flag = True )
#...............................................................................
@click.version_option(
  version = metadata('partis-pris')['Version'] )
#...............................................................................
@click.pass_context
def cli(
  ctx,
  verbosity,
  with_color,
  ascii ):

  init_logging(
    level = verbosity,
    filename = None,
    with_color = with_color,
    with_unicode = not ascii )

  ctx.obj = EnvSpec()

  ctx.meta['log'] = getLogger('partis.pris')

cli.add_command(add)