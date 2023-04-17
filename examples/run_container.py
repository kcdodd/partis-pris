from partis.pris.container import Overlay, Binding, Container
from pathlib import Path

MPI_ENV = [
  # OpenMPI:
  'OMPI_*',
  'PMIX_*',
  'OPAL_*',
  'ORTE_*',
  # MPICH:
  'MPIR_*',
  'PMI_*',
  'PMIX_*',
  'HYDRA_*',
  # Intel MPI:
  'I_MPI_*',
  # MVAPICH2:
  'MV2_*']

c = Container('ubuntu-22.04-base.sif')

c.run_wait(
  command = ['env'],
  app = 'venv',
  env_pass = MPI_ENV,
  bindings = [
    Binding(
      src = './venv',
      dst = '/venv',
      writable = True)])

