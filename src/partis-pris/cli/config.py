import click

from ..utils import norm_index

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@click.command()
#...............................................................................
@click.option(
  '--index',
  help = "Additional index used to resolve requirements.",
  type = str,
  multiple = True )
@click.option(
  '--no-index',
  help = "Removes existing indexes.",
  type = bool,
  default = False,
  is_flag = True )
#...............................................................................
@click.pass_context
def config(
  ctx,
  index,
  no_index ):

  log = ctx.obj.logger

  index = [norm_index(idx) for idx in index]