from collections.abc import (
  Iterable,
  Sequence )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Index:
  #-----------------------------------------------------------------------------
  def __init__(self, index):
    self.index = norm_uri(index)

  #-----------------------------------------------------------------------------
  def __str__(self):
    return self.index

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class EnvSpec:
  #-----------------------------------------------------------------------------
  def __init__(self,
    explicit = None,
    implicit = None ):

    if explicit is None:
      explicit = tuple()

    if implicit is None:
      implicit = tuple()

    self._explicit = tuple(explicit)
    self._implicit = tuple(implicit)

    self._staged_add = list()
    self._staged_rm = list()

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
