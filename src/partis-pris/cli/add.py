import sys
import os
from pathlib import Path
from urllib.parse import (
  quote as urlquote,
  urlparse )

import click


from partis.pyproj.pptoml import (
  pptoml as pptoml_type )
from partis.pyproj.validate import (
  validating)
from partis.utils import (
  getLogger,
  branched_log,
  checksum,
  ModelHint,
  ModelError,
  Loc,
  LogListHandler,
  VirtualEnv,
  MutexFile )

from ..utils import norm_req, norm_reqfile

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@click.command()
#...............................................................................
@click.option(
  '-r', '--requirement',
  help = "Add an explicit requirement (e.g. partis-pris==0.1.0)",
  type = str,
  multiple = True )
@click.option(
  '-f', '--requirement-file',
  help = "Add from requirements file (e.g. requirements.txt)",
  type = Path,
  multiple = True )
#...............................................................................
@click.option(
  '--build-deps',
  help = "Also add the build dependencies.",
  type = bool,
  default = False,
  is_flag = True )
@click.option(
  '--no-deps',
  help = "Don't add implicit dependencies.",
  type = bool,
  default = False,
  is_flag = True )
#...............................................................................
@click.pass_context
def add(
  ctx,
  requirement,
  requirement_file,
  build_deps,
  no_deps ):

  log = ctx.obj.logger

  reqs = [
    *[norm_req(req) for req in requirement],
    *[req for file in requirement_file for req in norm_reqfile(file)] ]

  # log.hint(ModelHint(
  #   "Specification",
  #   hints = [
  #     ModelHint(
  #       f"Explicit requirements ({len(requirement)})",
  #       hints = requirement) if requirement else ModelHint(
  #         'No requirements specified',
  #         level = 'warning'),
  #     ModelHint(
  #       f"Resolution Index ({len(index)})",
  #       hints = index) if index else ModelHint(
  #         'No index specified for resolution',
  #         level = 'error') ]))

  # if requirement and not index:
  #   raise click.Abort()

  spec = ctx.obj
  spec.add(reqs)
