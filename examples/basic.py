"""
basic.py — Two square panels side by side.

Layout
------
  ┌──────────┬──────────┐
  │  left    │  right   │
  └──────────┴──────────┘

Both columns are 'auto' with aspect=1, so the solver makes each one as
wide as it is tall.  The figure height is inferred automatically.
"""

import numpy as np
import matplotlib.pyplot as plt
from mplayout import Grid

rng = np.random.default_rng(0)

g = Grid(rows=['auto'], cols=['auto', 'auto'], gap='4mm', margin='5mm')
p_left  = g.panel(row=0, col=0, aspect=1.0)
p_right = g.panel(row=0, col=1, aspect=1.0)

fig, axes = g.build(fig_width='5in')

axes[p_left].imshow(rng.random((32, 32)), cmap='viridis', origin='lower')
axes[p_left].set_title('Left')

axes[p_right].imshow(rng.random((32, 32)), cmap='magma', origin='lower')
axes[p_right].set_title('Right')

fig.savefig('examples/basic.png', dpi=150)
plt.show()
