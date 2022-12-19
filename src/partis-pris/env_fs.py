import os
import sysconfig
from time import time
from enum import Enum
from pathlib import Path, PurePosixPath
import dataclasses
import typing
from fs.base import FS
from fs.osfs import OSFS
from fs.mountfs import MountFS
from fs.tarfs import ReadTarFS, WriteTarFS
from fs.walk import Walker
from fs.copy import copy_fs

import orjson

from partis.pyproj import b64_nopad

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class SchemeKey(Enum):
  platlib = 'platlib'
  purelib = 'purelib'
  headers = 'headers'
  scripts = 'scripts'
  data = 'data'

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@dataclasses.dataclass
class FileRef:
  pkg: str
  audited: float
  hash: str
  size: int

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@dataclasses.dataclass
class Scheme:
  platlib: Path
  purelib: Path
  headers: Path
  scripts: Path
  data: Path

  #-----------------------------------------------------------------------------
  @classmethod
  def from_sysconfig(cls, *args, **kwargs):
    paths = sysconfig.get_paths(*args, **kwargs)

    return Scheme(
      platlib = Path(paths['platlib']),
      purelib = Path(paths['purelib']),
      headers = Path(paths['include']),
      scripts = Path(paths['scripts']),
      data = Path(paths['data']) )

  #-----------------------------------------------------------------------------
  def fs(self):
    efs = MountFS()

    for s in SchemeKey:
      efs.mount(
        s.name,
        OSFS(os.fspath(getattr(self, s.value) )))

    return efs

  #-----------------------------------------------------------------------------
  def audit(self, manifest: Path):
    manifest_tmp = manifest.with_name(manifest.name + '.tmp')

    if manifest_tmp.exists():
      manifest_tmp.unlink()

    with WriteTarFS(os.fspath(manifest_tmp)) as mfs:

      # if manifest.exists():
      #   with ReadTarFS(os.fspath(manifest)) as _mfs:
      #     copy_fs(_mfs, mfs)

      if not mfs.exists('/RECORD'):
        mfs.makedir('/RECORD')

      rfs = mfs.opendir('/RECORD')

      for s in SchemeKey:
        if not rfs.exists(s.value):
          rfs.makedir(s.value)

      with self.fs() as envfs:
        dist_info = dict()

        purelib = envfs.opendir(f'/purelib')
        platlib = envfs.opendir(f'/platlib')

        dinfo = '.dist-info'
        for path, info in [*purelib.glob(f'*{dinfo}/'), *platlib.glob(f'*{dinfo}/')]:

          name, version = info.name[:-len(dinfo)].split('-')
          print(f'{name} {version} -> {path}')

          dist_info[name] = (version, path)

        walker = Walker(
          exclude_dirs = [
            '__pycache__'])

        libs = [
          f'/purelib/{lib}'
          for lib in purelib.listdir('/')
          if lib != '__pycache__']

        for lib in libs:

          if envfs.isfile(lib):
            print(f'{lib}')
            check_record(envfs, rfs, lib)

          else:
            for dir, dirs, files in walker.walk(envfs, lib):
              print(f'{dir}: {len(files)}')

              if not rfs.exists(dir):
                rfs.makedir(dir)

              for file in files:
                check_record(envfs, rfs, f'{dir}/{file.name}')

    manifest_tmp.replace(manifest)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def check_record(envfs, rfs, path):
  _record = None

  if rfs.exists(path):
    _record = FileRef(**orjson.loads(rfs.readbytes(path)))
    mtime = envfs.getmodified(path).timestamp()
    print(f'{_record.audited} > {mtime} = {_record.audited > mtime}')
    if _record.audited > mtime:
      return

  hash_hex = envfs.hash(path, 'sha256')
  hash = 'sha256=' + b64_nopad(bytes.fromhex(hash_hex))
  size = envfs.getsize(path)

  record = FileRef(
    pkg = '',
    audited = time(),
    hash = hash,
    size = size )

  if _record and (_record.hash != record.hash or _record.size != record.size):
    print(f"modified: {_record} -> {record}")

  rfs.touch(path)
  rfs.writebytes(path, orjson.dumps(record))

