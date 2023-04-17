from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import nullcontext
from functools import partial
from shlex import quote
import trio
import shutil
from partis.utils.async_trio import (
  wait_all )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
limit = trio.CapacityLimiter(100)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def total_size(path):
  if path.is_dir():
    return sum(total_size(p) for p in path.iterdir())

  return path.lstat().st_size

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def replace(src: Path, dst: Path):
  """An implementation of "replace" that doesn't fail on cross-device moves
  """
  dst = dst.resolve()
  dst_tmp = (dst.parent / (dst.name + '.move_tmp'))
  if dst_tmp.exists():
    dst_tmp.unlink()

  shutil.copy2(str(src), str(dst_tmp))
  dst_tmp.replace(dst)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Overlay:
  #-----------------------------------------------------------------------------
  def __init__(self,
      name: str,
      out_dir: Path = '.',
      size_mb: int = 64 ):

    self._name = name
    self._out_dir = Path(out_dir).resolve()
    self._overlay_img = self._out_dir/(self._name+'.img')
    self._size_mb = size_mb

  #-----------------------------------------------------------------------------
  async def build(self,
      tmp: Path = None ):

    if not self._overlay_img.exists():
      with nullcontext(tmp) if tmp else TemporaryDirectory() as tmp:
        await self._build(tmp)

    return self._overlay_img

  #-----------------------------------------------------------------------------
  async def _build(self, tmp: Path):
      tmp = Path(tmp)
      overlay_dir = tmp/self._name
      overlay_dir.mkdir()
      top = (overlay_dir/'upper'/self._name)
      top.mkdir(parents = True)

      (overlay_dir/'work').mkdir()

      overlay_img = tmp/(self._overlay_img.name)

      await trio.run_process(
        command = [
          'dd',
          'if=/dev/zero',
          f'of={overlay_img}',
          'bs=1M',
          f'count={self._size_mb:d}'])

      await trio.run_process(
        command = [
          'mkfs.ext3',
          '-d', str(overlay_dir),
          str(overlay_img)])

      self._out_dir.mkdir(parents = True, exist_ok = True)
      replace(overlay_img, self._overlay_img)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Binding:
  #-----------------------------------------------------------------------------
  def __init__(self,
      src: Path,
      dst: Path,
      writable: bool = False):

    self._src = Path(src)
    self._dst = Path(dst)
    self._writable = bool(writable)

    if not self._dst.is_absolute():
      raise ValueError(
        f"'dst' must be absolute: {self._dst}")

  #-----------------------------------------------------------------------------
  @property
  def bind(self):
    path = self._src.resolve()

    if not path.is_dir():
      raise ValueError(
        f"'src' must be an existing directory: {path}")

    opt = 'rw' if self._writable else 'ro'
    return f"{path}:{self._dst}:{opt}"

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Container:
  #-----------------------------------------------------------------------------
  def __init__(self,
      image: Path):
    self._image = Path(image).resolve()

  #-----------------------------------------------------------------------------
  async def run(self,
      command = [],
      app: str = None,
      overlays: list[Overlay] = [],
      bindings: list[Binding] = [],
      env_pass: list[str] = [],
      env: dict[str,str] = {}):

    with TemporaryDirectory() as tmp:
      tmp = Path(tmp)
      env_file = tmp/'env_file.txt'

      env_file.write_text('\n'.join([
        f"{k}='{quote(str(v))}'"
        for k,v in env.items()]))

      overlays = await wait_all([o.build(tmp) for o in overlays])

      await trio.run_process(
        command = [
          'singularity',
          'exec',
          *(['--app', app] if app else []),
          '--cleanenv',
          '--env', f"\"{','.join([str(v) for v in env_pass])}\"",
          '--env-file', str(env_file),
          '--no-home',
          *[c for o in overlays for c in ['--overlay', str(o)]],
          *[c for b in bindings for c in ['--bind', b.bind]],
          str(self._image),
          *command])

  #-----------------------------------------------------------------------------
  def run_wait(self, *args, **kwargs):
    trio.run(partial(self.run, *args, **kwargs))
