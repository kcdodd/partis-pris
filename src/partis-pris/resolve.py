r"""

A potential set of versions from project packages to be installed is
represented as an integer selection vector :math:`s_i`.
Projects are enumerated :math:`i \in \{0,1,2,... M\}`, and their :math:`N_i`
available package versions :math:`s_i \in \{0,1,2,... N_i\}`, where version :math:`0`
is a special value representing the "null" version, or *absence* of the package.
The selection vector can also be represented as an :math:`M \times N` binary matrix
:math:`S_{ij}` (:math:`N = max(N_i)`),

.. math::

  S_{ij} =
  \begin{cases}
    1, & j = s_i\\
    0, & \text{otherwise}
  \end{cases}.

Dependency constraints between packages are encoded in a binary incidence
matrix :math:`D_{ijkl}` (:math:`\{0,1\}^{M \times N} \rightarrow \{0,1\}^{M \times N}`),

.. math::

  D_{ijkl} =
  \begin{cases}
    1, & \text{(pkg. $k$, $l$) violates constrains of (pkg. $i$, $j$)}\\
    0, & \text{no violation}
  \end{cases},

that maps a subset of packages+version to another subset of packages+versions,
:math:`V_{ij} = D_{ijkl} S_{kl}`, whose constraints would be violated (if they were
installed together).

A selection of packages+versions is valid if it does not overlap with the set of
violations it induces,

.. math::

  S_{ij} V_{ij} = S_{ij} D_{ijkl} S_{kl} = 0.

This happens to look similar to the quadratic assignment problem, since a minimum
of this objective function corresponds to a valid selection (assuming there is at
least one), but its :

* :math:`S_{ij}` is a selection (or combinations with replacement) with
  :math:`\approx M^N` possible values, and *not* a permutation matrix.
* The "distance" matrix, and the resulting objective function, are boolean
  valued. There is no "better" or "worse" selection (it is either valid or not),
  so "greedy" algorithms do not apply.
* On the plus side, one local minimum is just as valid as any other selection.

So, while this way of representing dependency resolution does not seem to make the
problem fundamentally easier, the idea being explored is that the incidence matrix
(:math:`D_{ijkl}`) might be pre-computed and stored in some kind of optimized
form, and to enable more computationally efficient methods of finding a solution.

"""

import os
import os.path as osp
from pathlib import (
  Path,
  PurePosixPath)
from tempfile import TemporaryDirectory
import re
from email.parser import HeaderParser
from email.errors import MessageError
import tarfile, zipfile
import hashlib
from importlib import metadata
from dataclasses import (
  dataclass,
  field,
  asdict)
from typing import Optional
from subprocess import CalledProcessError

from pypi_simple import (
  PyPISimple,
  NoSuchProjectError)
import requests
from packaging.requirements import (
  Requirement,
  InvalidRequirement)
from packaging.version import (
  Version,
  InvalidVersion)
from packaging.tags import (
  Tag,
  parse_tag,
  sys_tags)
from packaging.specifiers import (
  SpecifierSet,
  InvalidSpecifier)
from packaging.markers import (
  Marker,
  UndefinedEnvironmentName,
  default_environment)
from packaging.utils import (
  parse_wheel_filename,
  InvalidWheelFilename)
import numpy as np

from orjson import (
  loads,
  dumps,
  JSONDecodeError)
import orjson
import tomli


from partis.pyproj import (
  ValidationError,
  validating,
  norm_dist_name,
  norm_dist_filename)

from partis.pyproj.pptoml import pptoml as PpToml

from partis.utils import (
  getLogger,
  LogListHandler,
  branched_log,
  ModelHint,
  VirtualEnv,
  MutexFile )
import logging

from pyproject_hooks import (
  BuildBackendHookCaller,
  HookMissing )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def prep(projects):
  npkg = len(projects)
  nv = 1 + max(len(proj.packages) for proj in projects)
  pnv = np.ones((npkg,), dtype = np.int32)

  # matrix encoding whether each version of each package violates a constraint
  mnot = np.zeros(
    ( npkg, nv, npkg, nv ),
    dtype = bool )

  pmap = { p.name : i for i, p in enumerate(projects) }

  # enumerate each version of the package.
  # verions 'zero' is reserved to represent the 'absence' of the package
  vmap = [
    { str(v.version): i+1 for i, v in enumerate(p.packages) }
    for p in projects ]

  for proj in projects:
    # enumerate all possible packages that could be installed
    pidx = pmap[proj.name]
    pnv[pidx] = 1 + len(proj.packages)

    for pkg in proj.packages:
      version = str(pkg.version)

      # get enuemrated index of each package version
      vidx = vmap[pidx][version]

      for dep in pkg.dependencies:
        # if not dep.marker.evaluate():
        #   continue

        if dep.name not in pmap:
          raise ValueError(f"Dependency of '{pkg.name}' not found: '{dep}'")

        # get enumerated indices for all dependencies of this package + version
        _pidx = pmap[dep.name]
        _pkg = projects[_pidx]

        # the absence of the dependency (version 'zero') violates constraint
        mnot[pidx, vidx, _pidx, 0] = True

        for _version, _vidx in vmap[_pidx].items():
          # compute whether this version violates a constraint
          compat = _version in dep.specifier
          mnot[pidx, vidx, _pidx, _vidx] = not compat

  return pmap, pnv, mnot

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def check_valid(mnot, p):

  # computes the image of constraint violations for a selection of package versions
  violators = np.einsum(
    'ijkl->kl',
    np.take_along_axis(
      mnot,
      p[:,None,None,None],
      axis = 1 ))

  # take only those violations which overlap with the selected versions
  # I.E. valid if the set of packages versions is disjoint with the induced set
  # of constrain violations
  pkg_violators = np.einsum(
    'ij->i',
    np.take_along_axis(
      violators,
      p[:,None],
      axis = 1 ))

  invalid = pkg_violators.any()
  valid = not invalid

  return valid, pkg_violators, violators

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def resolve(mnot, p0):
  p = np.copy(p)

  print('initial p', p)
  valid, pkg_violators, violators = check_valid(mnot, p)
  print('valid', valid)
  # print(violators)

  while not valid:
    # pkg_violators = violators.any(axis=1)

    pidxs = np.nonzero(pkg_violators)[0]
    print('pkg_violators', pidxs)
    print('violations', [ list(np.nonzero(v)[0]) for v in violators[pidxs] ])

    for pidx in pidxs:
      nv = pnv[pidx]
      print(f'pidx: {pidx}')
      # hypothesize that this package violation can be resolved, ignoring other packages
      m = (p != 0) & ~pkg_violators
      m[pidx] = True

      vold = p[pidx]

      for i in range(nv-1):
        vnew = (vold + nv - 1) % nv
        print(f'  {vold} -> {vnew}')
        _valid, _, _ = check_valid(mnot[m][:,:,m], p[m])
        assert not _valid

        p[pidx] = vnew

        # compute if this does not invalidate currently valid packages
        _valid, _, _ = check_valid(mnot[m][:,:,m], p[m])

        print(f'  -> {_valid}')

        if _valid:
          break

        vold = vnew

      else:
        # raise ValueError('failed')
        continue

      break

    else:
      raise ValueError('failed')

    print('p', p)
    valid, pkg_violators, violators = check_valid(mnot, p)
    print('valid', valid)

  print('install p', p)
  valid, pkg_violators, violators = check_valid(mnot, p)
  print('valid', valid)

