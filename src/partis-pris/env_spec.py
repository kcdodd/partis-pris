import sys
import os
from pathlib import Path
from collections.abc import (
  Iterable,
  Sequence )

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

from .env_fs import (
  Scheme )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Index:
  #-----------------------------------------------------------------------------
  def __init__(self, index):
    self.index = norm_uri(index)

  #-----------------------------------------------------------------------------
  def __str__(self):
    return self.index

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class EnvSpec(VirtualEnv):
  #-----------------------------------------------------------------------------
  def __init__(self,
    path = None,
    explicit = None,
    implicit = None,
    **kwargs ):

    super().__init__(
      path = path if path else Path(sys.executable).parent.parent,
      **kwargs )

    if explicit is None:
      explicit = tuple()

    if implicit is None:
      implicit = tuple()

    self._explicit = tuple(explicit)
    self._implicit = tuple(implicit)

    self._staged_add = list()
    self._staged_rm = list()

    self._scheme = Scheme.from_sysconfig()

  #-----------------------------------------------------------------------------
  def audit(self):
    self._scheme.audit(
      manifest = self.path / 'pris_manifest.tgz')

  #-----------------------------------------------------------------------------
  @property
  def explicit(self):
    return self._explicit

  #-----------------------------------------------------------------------------
  @property
  def implicit(self):
    return self._implicit

  #-----------------------------------------------------------------------------
  def add(self, reqs, index):
    self._staged_add.extend(reqs)

  #-----------------------------------------------------------------------------
  def rm(self, reqs):
    self._staged_rm.extend(reqs)

