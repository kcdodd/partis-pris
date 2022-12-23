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
from importlib import metadata
import dataclasses

from pypi_simple import PyPISimple, NoSuchProjectError, tqdm_progress_factory

from packaging.requirements import Requirement, InvalidRequirement
from packaging.version import Version, InvalidVersion
from packaging.tags import Tag, parse_tag, sys_tags
from packaging.specifiers import SpecifierSet, InvalidSpecifier
from packaging.markers import Marker, UndefinedEnvironmentName, default_environment
from packaging.utils import parse_wheel_filename, InvalidWheelFilename

import numpy as np
from orjson import loads, dumps, JSONDecodeError
import orjson

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
pkg_info_rec = re.compile(r'[\w.-]+/PKG-INFO')
metadata_rec = re.compile(r'[\w.-]+\.dist-info/METADATA')

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@dataclasses.dataclass
class IndexedPackage:
  name: str
  version: Version
  dist: str
  filename: str
  url: str
  digests: dict[str,str]

  #-----------------------------------------------------------------------------
  def __post_init__(self):
    self.version = self.version if isinstance(self.version, Version) else Version(self.version)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@dataclasses.dataclass
class IndexedProject:
  name: str
  packages: list[IndexedPackage]

  #-----------------------------------------------------------------------------
  def __post_init__(self):
    self.packages = [
      v if isinstance(v, IndexedPackage) else IndexedPackage(**v)
      for v in self.packages ]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@dataclasses.dataclass
class Package:
  name: str
  version: Version
  dist: str
  file: Path
  url: str
  digests: dict[str,str]
  reqs: list[Requirement]

  #-----------------------------------------------------------------------------
  def __post_init__(self):
    self.version = self.version if isinstance(self.version, Version) else Version(self.version)
    self.file = Path(self.file)
    self.reqs = [ r if isinstance(r, Requirement) else Requirement(r) for r in self.reqs ]


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def dumper(obj):
  if isinstance(obj, (Version, Requirement, Path)):
    return str(obj)

  raise TypeError

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Project:
  #-----------------------------------------------------------------------------
  def __init__(self, name, packages):
    self.name = name
    self.packages = packages

    self.versions = {
      v : pkg.reqs
      for v, pkg in self.packages.items() }

    # enumerate each version of the package.
    # verions 'zero' is reserved to represent the 'absence' of the package
    self.vmap = { v: i+1 for i, v in enumerate(self.versions.keys()) }

  #-----------------------------------------------------------------------------
  def __hash__(self):
    return hash(self.name)

  #-----------------------------------------------------------------------------
  def __eq__(self, other):
    return self.name == other.name

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class MetadataNotFoundError(FileNotFoundError):
  pass

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class JsonBackedDict(dict):
  #-----------------------------------------------------------------------------
  def __init__(self, file):
    self.file = Path(file)

  #-----------------------------------------------------------------------------
  def __enter__(self):
    if self.file.exists():
      try:
        with open(self.file, 'rb') as fp:
          self.update(loads(fp.read()))
      except JSONDecodeError as e:
        self.file.unlink()

    return self

  #-----------------------------------------------------------------------------
  def __exit__(self, type, value, trace):
    if type is None:
      with open(self.file, 'wb') as fp:
        fp.write(dumps(self, default = dumper, option = orjson.OPT_INDENT_2))

    self.clear()
    return False

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

    self.manifest = JsonBackedDict(file = self.cache_dir / 'manifest.json')
    self.indexed = JsonBackedDict(file = self.cache_dir / 'indexed.json')

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
    pkg_types = None,
    environment = None,
    tags = None,
    prereleases = False,
    devreleases = False ):

    if isinstance(requirements, (str, Requirement)):
      requirements = [requirements]

    if environment is None:
      environment = default_environment()

    projects = list()

    if pkg_types is None:
      pkg_types = [ 'wheel', 'sdist' ]

    if 'wheel' in pkg_types:
      if tags is None:
        tags = sys_tags()

      tags = set(tags)

    # print(f"For tags: {[ str(t) for t in tags ]}")

    with self.indexed as indexed, self.manifest as manifest:
      for req in requirements:
        req = req if isinstance(req, Requirement) else Requirement(req)

        if req.marker and not req.marker.evaluate(environment):
          continue

        name = req.name
        pkg_manifest = manifest.setdefault(name, {})

        pkg_cache_dir = self.cache_dir / 'packages' / name
        pkg_cache_dir.mkdir(parents = True, exist_ok = True)

        if name in indexed:
          project = IndexedProject(**indexed[name])

        else:
          try:
            # TODO: handle retry multiple remotes, or from locals
            for remote in self.remotes:
              with PyPISimple(**remote) as client:
                project = client.get_project_page(name)

          except NoSuchProjectError:
            print(f"{name}: no index, skipping...")
            continue

          packages = list()

          for pkg in project.packages:
            if pkg.is_yanked:
              continue

            try:
              version = Version(pkg.version)
            except InvalidVersion as e:
              print(f'  - error: {pkg.filename} -> {e}')
              continue

            packages.append(IndexedPackage(
                name = name,
                version = pkg.version,
                dist = pkg.package_type,
                filename = pkg.filename,
                url = pkg.url,
                digests = pkg.digests))

          project = IndexedProject(
            name = name,
            packages = packages)

          indexed[name] = project

        print(f"{name}: {len(project.packages)}")

        packages = {}

        for pkg in project.packages:

          version = pkg.version

          if not version:
            continue

          if pkg.dist not in pkg_types:
            continue

          if pkg.dist == 'wheel':
            try:
              _,_,_, pkg_tags = parse_wheel_filename(pkg.filename)
            except InvalidWheelFilename as e:
              print(f'  - error: {pkg.filename} -> {e}')
              continue

            for tag in tags:
              if tag in pkg_tags:
                break

            else:
              print(f'  - no tags: {[ str(t) for t in pkg_tags ]}')
              continue

          if not prereleases and version.is_prerelease:
            continue

          if not devreleases and version.is_devrelease:
            continue

          if version in packages:
            continue

          if req.specifier and version not in req.specifier:
            print(f'  - not in specifier: {pkg.filename} -> {version}')
            continue

          _version = str(version)

          dl_manifest = pkg_manifest.setdefault(_version, {}).setdefault(pkg.dist, {})

          if pkg.filename in dl_manifest:
            packages[version] = Package(**dl_manifest[pkg.filename])
            continue

          dl_dir = pkg_cache_dir / str(version) / pkg.dist
          dl_dir.mkdir(parents = True, exist_ok = True)
          file = dl_dir / pkg.filename
          file_tmp = self.tmp_cache_dir / pkg.filename

          # fixup
          # _dl_dir = pkg_cache_dir / str(version) / pkg.dist
          # _dl_dir.mkdir(parents = True, exist_ok = True)
          # _file = dl_dir / pkg.filename
          # if _file.exists():
          #   _file.replace(file)

          method = 'cache' if file.exists() else 'download'
          print(f"  - {name}, {version}: ({method}) {file.name}")

          if not file.exists():
            try:
              with PyPISimple(**remote) as client:
                client.download_package(
                  pkg,
                  file_tmp,
                  progress = tqdm_progress_factory())

              file_tmp.replace(file)

            except:
              if file_tmp.exists():
                file_tmp.unlink()

              raise

          buf = None

          try:
            if pkg.dist == 'sdist':

              if file.suffix == '.zip':
                with zipfile.ZipFile(file, mode = 'r') as fp:
                  for aname in fp.namelist():
                    if pkg_info_rec.fullmatch(aname):
                      buf = fp.read(aname)
                      break
                  else:
                    raise MetadataNotFoundError(f'PKG-INFO not found: {file.name}')

              else:
                with tarfile.open(file, mode = 'r:*', format = tarfile.PAX_FORMAT) as fp:
                  for aname in fp.getnames():
                    if pkg_info_rec.fullmatch(aname):
                      buf = fp.extractfile(aname).read()
                      break
                  else:
                    raise MetadataNotFoundError(f'PKG-INFO not found: {file.name}')
            else:
              # wheel
              with zipfile.ZipFile(file, mode = 'r') as fp:
                for aname in fp.namelist():
                  if metadata_rec.fullmatch(aname):
                    buf = fp.read(aname)
                    break
                else:
                  raise MetadataNotFoundError(f'METADATA not found: {file.name}')

          except (zipfile.BadZipFile, tarfile.TarError, MetadataNotFoundError) as e:
            print(f'    - error: {e}')
            file.unlink()
            continue

          try:
            txt = buf.decode('utf-8', errors = 'strict')
            py_req, reqs = parse_core_metadata_reqs(txt)
            reqs = [Requirement(r) for r in reqs]

            if py_req:
              if Version(environment['python_version']) not in SpecifierSet(py_req):
                continue

          except (UnicodeError, MessageError, InvalidRequirement, InvalidSpecifier) as e:
            print(f'    - error: {e}')
            file.unlink()
            continue

          print(f'    - requirements: {len(reqs)}')

          pkg = Package(
            name = name,
            version = version,
            file = file,
            dist = pkg.dist,
            url = pkg.url,
            digests = pkg.digests,
            reqs = reqs )

          dl_manifest[pkg.file.name] = pkg
          packages[version] = pkg

        projects.append(Project(
          name = name,
          packages = packages))

    return projects


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def prep(projects):
  npkg = len(projects)
  nv = 1 + max(len(pkg.versions) for pkg in projects)
  pnv = np.ones((npkg,), dtype = np.int32)

  # matrix encoding whether each version of each package violates a constraint
  mnot = np.zeros(
    ( npkg, nv, npkg, nv ),
    dtype = bool )

  pmap = { p.name : i for i, p in enumerate(projects) }

  for pkg in projects:
    # enumerate all possible packages that could be installed
    pidx = pmap[pkg.name]
    pnv[pidx] = 1 + len(pkg.versions)

    for version, deps in pkg.versions.items():
      # get enuemrated index of each package version
      vidx = pkg.vmap[version]

      for dep in deps:
        # if not dep.marker.evaluate():
        #   continue

        if dep.name not in pmap:
          raise ValueError(f"Dependency of '{pkg.name}' not found: '{dep}'")

        # get enumerated indices for all dependencies of this package + version
        _pidx = pmap[dep.name]
        _pkg = projects[_pidx]

        # the absence of the dependency (version 'zero') violates constraint
        mnot[pidx, vidx, _pidx, 0] = True

        for _version, _vidx in _pkg.vmap.items():
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
