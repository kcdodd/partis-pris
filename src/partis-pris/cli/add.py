"""Install things
"""

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
import sys
import os
from pathlib import Path
from urllib.parse import (
  quote as urlquote,
  urlparse )

import click
import tomli

from packaging.requirements import (
  Requirement,
  InvalidRequirement )
from packaging.specifiers import SpecifierSet

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

DEFAULT_INDEX = 'https://pypi.org/simple'

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def norm_uri(uri, cwd = None):
  parts = urlparse(uri)

  if len(parts.scheme) <= 1:
    # assume the uri is actually a local file
    path = Path(uri)

    if not path.is_absolute():
      path = (Path.cwd() if cwd is None else Path(cwd)) / path

    uri = path.as_uri()

  return uri

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def norm_index(index):
  return norm_uri(index)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def _norm_index(ctx, param, value):
  return [norm_index(index) for index in value]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def norm_req(req):

  try:
    return Requirement(req)

  except InvalidRequirement as e:
    req = req.strip()

    if '/' in req or '\\' in req:
      extras = ''

      if req.endswith(']'):
        parts = req[:-1].split('[')

        if len(parts) > 0:
          extras = f'[{parts[-1]}]'
          parts = parts[:-1]

        req = '['.join(parts)

      uri = norm_uri(req)

      return Requirement(f'direct_reference{extras}@ {uri}')

    raise

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def _norm_req(ctx, param, value):
  return [norm_req(req) for req in value]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def _norm_req_file(ctx, param, value):

  reqs = list()

  for file in value:
    with open(file, 'rb') as fp:
      src = fp.read()
      src = src.decode( 'utf-8', errors = 'replace' )

      for line in src.splitlines():
        line = line.strip()

        if line.startswith('#'):
          pass
        elif line.startswith('-'):
          # TODO: unsupported pip options
          pass
        else:
          reqs.append(norm_req(line))

  return reqs

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@click.command()
#...............................................................................
@click.option(
  '-r', '--requirement',
  help = "Add an explicit requirement (e.g. partis-pris==0.1.0)",
  type = str,
  multiple = True,
  callback = _norm_req)
@click.option(
  '-f', '--requirement-file',
  help = "Add from requirements file (e.g. requirements.txt)",
  type = Path,
  multiple = True,
  callback = _norm_req_file)
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
@click.option(
  '--index',
  help = "Additional index used to resolve requirements.",
  type = str,
  multiple = True,
  callback = _norm_index)
@click.option(
  '--no-index',
  help = "Disable the default index.",
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
  no_deps,
  index,
  no_index ):

  log = ctx.meta['log']

  if not no_index:
    index.append(norm_uri(DEFAULT_INDEX))

  requirement.extend(requirement_file)

  log.hint(ModelHint(
    "Specification",
    hints = [
      ModelHint(
        f"Explicit requirements ({len(requirement)})",
        hints = requirement) if requirement else ModelHint(
          'No requirements specified',
          level = 'warning'),
      ModelHint(
        f"Resolution Index ({len(index)})",
        hints = index) if index else ModelHint(
          'No index specified for resolution',
          level = 'error') ]))

  if requirement and not index:
    raise click.Abort()

  spec = ctx.obj
  spec.add(requirement, index)
