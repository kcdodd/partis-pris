r"""

The vector :math:`p_i` represents a set of project packages and versions to be
installed.
Projects are enumerated :math:`i \in \{0,1,2,... M\}`, and their :math:`N_i`
available versions :math:`p_i \in \{0,1,2,... N_i\}`, where :math:`p_i = 0`
represents the absence of the package.

.. math::

  P_{ij} =
  \begin{cases}
    1, & j = p_i\\
    0, & \text{otherwise}
  \end{cases}.

Dependencies and constraints between packages are encoded in a matrix that
maps a set of packages+version to the set of packages+versions that violate
constraints from package inter-dependencies, or a requested installation
(self-imposed constraints).
The mapping is :math:`\{0,1\}^{M \times N} \rightarrow \{0,1\}^{M \times N}`
(:math:`N = max(N_i)`)

.. math::

  V_{ijkl} =
  \begin{cases}
    1, & \text{(pkg. $k$ ver. $l$) violates constrains from (pkg. $i$ ver. $j$)}\\
    0, & \text{no violation}
  \end{cases}

A set of packages+versions is valid if the set of violations does
not overlap.

.. math::

  \operatorname{Tr}\left(P^{T}VP\right) = 0

"""

import os
import os.path as osp
from pathlib import Path
import re
from email.parser import HeaderParser
from email.errors import MessageError
import tarfile, zipfile
from importlib import metadata
from pypi_simple import PyPISimple, NoSuchProjectError, tqdm_progress_factory

from packaging.requirements import Requirement, InvalidRequirement
from packaging.version import Version, InvalidVersion
from packaging.tags import Tag, parse_tag, sys_tags
from packaging.specifiers import SpecifierSet, InvalidSpecifier
from packaging.markers import Marker, UndefinedEnvironmentName, default_environment
from packaging.utils import parse_wheel_filename, InvalidWheelFilename

import numpy as np

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
pkg_info_rec = re.compile(r'[\w.-]+/PKG-INFO')
metadata_rec = re.compile(r'[\w.-]+\.dist-info/METADATA')

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Project:
  #-----------------------------------------------------------------------------
  def __init__(self, name, packages):
    self.name = name
    self.packages = packages

    self.versions = {
      v : [
        req if isinstance(req, Requirement) else Requirement(req)
        for req in (list(x[2]) if isinstance(x, tuple) else (x if x else list())) ]
      for v, x in self.packages.items() }

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

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class MetadataNotFoundError(FileNotFoundError):
  pass

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

    for req in requirements:
      req = req if isinstance(req, Requirement) else Requirement(req)

      if req.marker and not req.marker.evaluate(environment):
        continue

      name = req.name

      pkg_cache_dir = self.cache_dir / 'packages' / name
      pkg_cache_dir.mkdir(parents = True, exist_ok = True)

      try:
        # TODO: handle retry multiple remotes, or from locals
        for remote in self.remotes:
          with PyPISimple(**remote) as client:
            project = client.get_project_page(name)

      except NoSuchProjectError:
        print(f"{name}: no index, skipping...")
        continue

      print(f"{name}: {len(project.packages)}")

      packages = {}

      for pkg in project.packages:

        version = pkg.version

        if not version:
          continue

        if pkg.is_yanked or pkg.package_type not in pkg_types:
          continue

        if pkg.package_type == 'wheel':
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

        try:
          version = Version(version)
        except InvalidVersion as e:
          print(f'  - error: {pkg.filename} -> {e}')
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

        dl_dir = pkg_cache_dir / str(version) / pkg.package_type
        dl_dir.mkdir(parents = True, exist_ok = True)
        file = dl_dir / pkg.filename
        file_tmp = self.tmp_cache_dir / pkg.filename

        # fixup
        # _dl_dir = pkg_cache_dir / str(version) / pkg.package_type
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
          if pkg.package_type == 'sdist':

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

        packages[version] = (pkg, file, reqs)

      projects.append(Project(
        name = name,
        packages = packages))

    return projects


