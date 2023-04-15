from partis.pris.container import Overlay, Binding, Container
from pathlib import Path

c = Container(Path('ubuntu-22.04-base.sif'))
c.run_wait(
  command = ['env'],
  env = {'FOO': 'BAR'},
  overlays = [Overlay('test_overlay')],
  bindings = [Binding('../installation', '/installation')])

