import sys
import os
from pathlib import Path
from urllib.parse import (
  quote as urlquote,
  urlparse )


from packaging.requirements import (
  Requirement,
  InvalidRequirement )

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

      return Requirement(f'direct_url{extras}@ {uri}')

    raise

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def norm_reqfile(reqfile):

  reqs = list()

  with open(reqfile, 'rb') as fp:
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