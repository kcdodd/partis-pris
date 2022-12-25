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
from pathlib import Path
import re
from email.parser import HeaderParser
from email.errors import MessageError
import tarfile, zipfile
import hashlib
from importlib import metadata
from dataclasses import dataclass, field, asdict
from typing import Optional

from pypi_simple import PyPISimple, NoSuchProjectError, tqdm_progress_factory
import requests

from packaging.requirements import Requirement, InvalidRequirement
from packaging.version import Version, InvalidVersion
from packaging.tags import Tag, parse_tag, sys_tags
from packaging.specifiers import SpecifierSet, InvalidSpecifier
from packaging.markers import Marker, UndefinedEnvironmentName, default_environment
from packaging.utils import parse_wheel_filename, InvalidWheelFilename

import numpy as np
from orjson import loads, dumps, JSONDecodeError
import orjson

from partis.pyproj import (
  ValidationError,
  norm_dist_name,
  norm_dist_filename)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
pkg_info_rec = re.compile(r'[\w.-]+/PKG-INFO')
metadata_rec = re.compile(r'[\w.-]+\.dist-info/METADATA')

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @dataclass
# class IndexedRequirement:
#   name: str
#   url: str
#   extras: set[str]
#   specifier: SpecifierSet
#   marker: Optional[Marker]

#   #-----------------------------------------------------------------------------
#   def __post_init__(self):
#     self.extras = set(self.extras)

#     if not isinstance(self.specifier, SpecifierSet):
#       self.specifier = SpecifierSet(self.specifier)

#     if self.marker and not isinstance(self.marker, Marker)
#       self.marker = Marker.__new__(Marker)
#       self.marker._markers = self.marker

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@dataclass
class IndexedPackage:
  name: str
  version: Version
  dist: str
  filename: str
  url: str
  digests: dict[str,str]

  # only for binary distributions
  build_number: tuple[int,str]
  tags: set[Tag]

  #-----------------------------------------------------------------------------
  def __post_init__(self):
    self.name = norm_dist_filename(norm_dist_name(self.name))

    try:
      self.version = (
        self.version if isinstance(self.version, Version)
        else Version(self.version) )

    except InvalidVersion as e:
      raise ValidationError(str(e)) from e

    if self.dist == 'wheel':
      if not self.tags:
        try:
          _,_, self.build_number, self.tags = parse_wheel_filename(self.filename)

        except InvalidWheelFilename as e:
          raise ValidationError(str(e)) from e

      else:
        self.build_number = tuple(self.build_number)
        self.tags = set([t if isinstance(t, Tag) else Tag(*t) for t in self.tags])

    else:
      self.build_number = ()
      self.tags = set()

  #-----------------------------------------------------------------------------
  def __str__(self):
    return f"{self.name}-{self.version}"

  #-----------------------------------------------------------------------------
  def __hash__(self):
    return hash(str(self))

  #-----------------------------------------------------------------------------
  def __eq__(self, other):
    return str(self) == str(other)

  #-----------------------------------------------------------------------------
  def download(self, file, timeout = None):
    file.parent.mkdir(parents = True, exist_ok = True)
    file_tmp = file.parent / (file.name + '.tmp')

    r_kwargs = dict(
      url = self.url,
      stream = True,
      timeout = timeout )

    for alg, digest_ref in self.digests.items():
      try:
        hasher = hashlib.new(alg)
        break
      except ValueError:
        continue

    else:
      if self.digests:
        raise ValidationError(f"Digests not supported: {list(self.digests.keys())}")

      digest_ref = None
      alg = 'sha256'
      hasher = hashlib.sha256()

    size = 0
    bufsize = 2**16

    try:
      with requests.get(**r_kwargs) as r, file_tmp.open("wb") as fp:
        for buf in r.iter_content(bufsize):
          fp.write(buf)
          hasher.update(buf)
          size += len(buf)

      digest = hasher.hexdigest()
      file_tmp.replace(file)

      if digest_ref and digest != digest_ref:
        raise ValidationError(f"Invalid digest: {digest} != {digest_ref}")

      return

    except:
      if file_tmp.exists():
        file_tmp.unlink()

      raise

  #-----------------------------------------------------------------------------
  def inspect(self, file, timeout = None):

    buf = None

    try:
      if self.dist == 'sdist':

        if file.suffix == '.zip':
          with zipfile.ZipFile(file, mode = 'r') as fp:
            for aname in fp.namelist():
              if aname.count('/') == 1 and pkg_info_rec.fullmatch(aname):
                buf = fp.read(aname)
                break
            else:
              raise MetadataNotFoundError(f'PKG-INFO not found: {file.name}')

        else:
          with tarfile.open(file, mode = 'r:*', format = tarfile.PAX_FORMAT) as fp:
            for aname in fp.getnames():
              if aname.count('/') == 1 and pkg_info_rec.fullmatch(aname):
                buf = fp.extractfile(aname).read()
                break
            else:
              raise MetadataNotFoundError(f'PKG-INFO not found: {file.name}')
      else:
        # wheel
        with zipfile.ZipFile(file, mode = 'r') as fp:
          for aname in fp.namelist():
            if aname.count('/') == 1 and metadata_rec.fullmatch(aname):
              buf = fp.read(aname)
              break
          else:
            raise MetadataNotFoundError(f'METADATA not found: {file.name}')

    except (zipfile.BadZipFile, tarfile.TarError, MetadataNotFoundError) as e:
      raise ValidationError(str(e)) from e

    try:
      txt = buf.decode('utf-8', errors = 'strict')
      py_req, reqs = parse_core_metadata_reqs(txt)

      return Package(
        **asdict(self),
        file = file,
        requires_python = py_req,
        dependencies = reqs )

    except (ValidationError, UnicodeError, MessageError, InvalidRequirement, InvalidSpecifier) as e:
      raise ValidationError(str(e)) from e

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@dataclass
class Package(IndexedPackage):
  file: Path
  requires_python: SpecifierSet
  dependencies: list[Requirement]

  #-----------------------------------------------------------------------------
  def __post_init__(self):
    super().__post_init__()

    try:
      self.file = Path(self.file)

      if self.requires_python is None:
        self.requires_python = SpecifierSet()

      else:
        self.requires_python = (
          self.requires_python if isinstance(self.requires_python, SpecifierSet)
          else SpecifierSet(self.requires_python) )

      self.dependencies = [
        r if isinstance(r, Requirement) else Requirement(r)
        for r in self.dependencies ]

    except (ValueError, InvalidRequirement, InvalidSpecifier) as e:
      raise ValidationError(str(e)) from e

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@dataclass
class IndexedProject:
  name: str
  packages: list[IndexedPackage]

  #-----------------------------------------------------------------------------
  def __post_init__(self):
    self.packages = [
      v if isinstance(v, IndexedPackage) else IndexedPackage(**v)
      for v in self.packages ]

  #-----------------------------------------------------------------------------
  def __str__(self):
    return self.name

  #-----------------------------------------------------------------------------
  def __hash__(self):
    return hash(str(self))

  #-----------------------------------------------------------------------------
  def __eq__(self, other):
    return str(self) == str(other)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@dataclass
class Project(IndexedProject):
  packages: list[Package]

  #-----------------------------------------------------------------------------
  def __post_init__(self):

    packages = [
      v if isinstance(v, Package) else Package(**(
        asdict(v) if isinstance(v, IndexedProject) else v))
      for v in self.packages ]

    self.packages = sorted( packages, key = lambda v: v.version )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class MetadataNotFoundError(FileNotFoundError):
  pass

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class JsonBackedDict(dict):
  #-----------------------------------------------------------------------------
  def __init__(self, file, errors_clear = False):
    self.file = Path(file).resolve()
    self.errors_clear = errors_clear

  #-----------------------------------------------------------------------------
  def pull(self):
    if self.file.exists():
      try:
        with open(self.file, 'rb') as fp:
          self.update(loads(fp.read()))
      except JSONDecodeError as e:
        # if failed, remove backing file to start over
        if self.errors_clear:
          self.file.unlink()
        else:
          raise

  #-----------------------------------------------------------------------------
  def push(self):
    buf = dumps(self, default = json_prep, option = orjson.OPT_INDENT_2)

    try:
      file_tmp = self.file.parent / (self.file.name + '.tmp')

      with open(file_tmp, 'wb') as fp:
        fp.write(buf)

      file_tmp.replace(self.file)

    except BaseException as e:
      if file_tmp.exists():
        file_tmp.unlink()

      if not self.errors_clear:
        raise

  #-----------------------------------------------------------------------------
  def __enter__(self):
    self.pull()
    return self

  #-----------------------------------------------------------------------------
  def __exit__(self, type, value, trace):
    if type is None:
      # only save to backing file if there was no error to prevent saving
      # potentially inconsistent state
      self.push()

    return False

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def json_prep(obj):

  if isinstance(obj, (Version, Requirement, SpecifierSet, Path)):
    return str(obj)

  if isinstance(obj, (set, frozenset)):
    return list(obj)

  if isinstance(obj, Tag):
    return [obj.interpreter, obj.abi, obj.platform]

  if isinstance(obj, Marker):
    return list(obj._markers)

  raise TypeError

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class PackageIndex:
  #-----------------------------------------------------------------------------
  def __init__( self,
    cache_dir,
    remotes = None ):

    if remotes is None:
      remotes = [{}]

    self._cache_dir = Path(cache_dir)
    self._remotes = remotes

    self.cache_dir.mkdir(parents = True, exist_ok = True)

    self._tmp_cache_dir = cache_dir / 'tmp'
    self.tmp_cache_dir.mkdir(exist_ok = True)

  #-----------------------------------------------------------------------------
  @property
  def cache_dir(self):
    return self._cache_dir

  #-----------------------------------------------------------------------------
  @property
  def tmp_cache_dir(self):
    return self._tmp_cache_dir

  #-----------------------------------------------------------------------------
  @property
  def remotes(self):
    return self._remotes

  #-----------------------------------------------------------------------------
  def query(self,
    requirements,
    dists = None,
    environment = None,
    tags = None,
    prereleases = False,
    devreleases = False,
    top = None,
    refresh = False,
    timeout = None):

    if isinstance(requirements, (str, Requirement)):
      requirements = [requirements]

    if environment is None:
      environment = default_environment()

    py_version = Version(environment['python_version'])

    query_projects = list()

    if dists is None:
      dists = [ 'wheel', 'sdist' ]

    if 'wheel' in dists:
      if tags is None:
        tags = sys_tags()

      tags = set(tags)

    for req in requirements:
      req = req if isinstance(req, Requirement) else Requirement(req)

      if req.marker and not req.marker.evaluate(environment):
        continue

      name = norm_dist_filename(norm_dist_name(req.name))

      pkg_cache_dir = self.cache_dir / 'packages' / name
      pkg_cache_dir.mkdir(parents = True, exist_ok = True)

      # caches info for all packages, whether or not they are downloaded
      indexed = JsonBackedDict(
        file = pkg_cache_dir / 'indexed.json',
        errors_clear = True)

      # stores info for those packages that have been downloaed and analyzed
      manifest = JsonBackedDict(
        file = pkg_cache_dir / 'manifest.json',
        errors_clear = True )

      with indexed, manifest:

        if name in indexed and not refresh:
          # get from cached index package info
          idx_proj = IndexedProject(**indexed[name])

        else:
          try:
            # TODO: handle retry multiple remotes, or from locals
            for remote in self.remotes:
              with PyPISimple(**remote) as client:
                idx_proj = client.get_project_page(name)

          except NoSuchProjectError:
            print(f"{name}: no index, skipping...")
            continue

          # pypi_simple.DistributionPackage -> IndexPackage
          idx_packages = list()

          for pkg in idx_proj.packages:
            if pkg.is_yanked:
              # ??
              continue

            if not pkg.version:
              # version couldn't be parsed from filename
              continue

            try:
              idx_packages.append(IndexedPackage(
                name = name,
                version = pkg.version,
                dist = pkg.package_type,
                filename = pkg.filename,
                build_number = None,
                tags = None,
                url = pkg.url,
                digests = pkg.digests))

            except ValidationError as e:
              # ignore these, probably indicates there are more issues
              print(f'  - error: {pkg.filename} -> {e}')
              continue

          idx_proj = IndexedProject(
            name = name,
            packages = idx_packages)

          indexed[name] = idx_proj

        print(f"{name}: {len(idx_proj.packages)}")

        # filter by dist type
        query_dists = [
          pkg
          for pkg in idx_proj.packages
          if pkg.dist in dists ]

        # NOTE: _dists is reversed so it is in *increasing* priority
        _dists = dists[::-1]

        # sort by dist type, and then by build number, so that packages are
        # checked in the order of priority
        query_dists = sorted(
          query_dists,
          key = lambda v: ( v.version, _dists.index(v.dist), v.build_number ),
          reverse = True )

        # Only analyze those of interest IndexPackage -> Package
        query_packages = list()
        query_versions = set()

        for pkg in query_dists:
          version = pkg.version

          if version in query_versions:
            # only keep the first package found for each desired version
            continue

          if not prereleases and version.is_prerelease:
            continue

          if not devreleases and version.is_devrelease:
            continue

          if req.specifier and version not in req.specifier:
            print(f'  - not specifier: {pkg.filename} -> {version}')
            continue

          if pkg.dist == 'wheel':
            # For binary distributions, make sure the tags
            # accept if *any* package tag matches *any* desired tag
            for tag in pkg.tags:
              if tag in tags:
                break

            else:
              print(f'  - not tags: {pkg.filename} -> {[ str(t) for t in pkg.tags ]}')
              continue

          _version = str(version)

          _manifest = manifest.setdefault(_version, {}).setdefault(pkg.dist, {})

          dl_dir = pkg_cache_dir / str(version) / pkg.dist
          dl_dir.mkdir(parents = True, exist_ok = True)
          file = dl_dir / pkg.filename

          method = 'manifest' if pkg.filename in _manifest else ('cache' if file.exists() else 'download')
          print(f"  - {name}, {version}: ({method}) {file.name}")\

          if not file.exists():
            pkg.download(file, timeout = timeout)

          try:
            if pkg.filename in _manifest:
              pkg = Package(**_manifest[pkg.filename])

            else:
              pkg = pkg.inspect(file, timeout = timeout)
              _manifest[pkg.file.name] = pkg

          except ValidationError as e:
            print(f'    - invalid: {e}')

            _manifest.pop(pkg.filename, None)
            file.unlink()
            continue

          if pkg.requires_python and py_version not in pkg.requires_python:
            print(f'    - not python: {pkg.requires_python}')
            continue

          print(f'    - requirements: {len(pkg.dependencies)}')
          query_versions.add(version)
          query_packages.append(pkg)

          if top and len(query_versions) >= top:
            break

        query_projects.append(Project(
          name = name,
          packages = query_packages))

    return query_projects


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


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def parse_core_metadata_reqs(txt):
  msg = HeaderParser().parsestr(txt)
  reqs = msg.get_all('Requires-Dist') or list()
  py_req = msg.get('Requires-Python')
  return py_req, reqs
