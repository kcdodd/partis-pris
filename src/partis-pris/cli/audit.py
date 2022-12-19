
import click


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@click.command()
@click.pass_context
def audit(
  ctx ):

  ctx.obj.audit()