# mplayout

Constraint-based figure layout for matplotlib.

Define rows and columns with fixed, fractional, or aspect-ratio-driven sizes.
The layout is solved as a linear system before any axes are created, so every
panel lands exactly where you specify — no post-hoc `tight_layout` surprises.

## Installation

```bash
pip install mplayout
```

Dependencies: `matplotlib`, `numpy`.

## Size specs

| Spec | Meaning |
|---|---|
| `1.5` / `'1.5in'` / `'15mm'` / `'1cm'` | Fixed size in inches (bare numbers are inches) |
| `'1fr'` / `'2fr'` | Proportional share of the remaining space (requires `fig_height` when used in rows) |
| `'auto'` | Determined by the aspect-ratio constraint of the panel(s) in that track |

## Quick start

```python
from mplayout import Grid

g = Grid(
    rows=['auto', '1fr'],
    cols=['auto', '5mm', 'auto'],
    hgap='2mm', wgap='2mm', margin='5mm',
)

p_img_left  = g.panel(row=0, col=0, aspect=1.0)
p_img_right = g.panel(row=0, col=2, aspect=4/3)
p_data      = g.panel(row=1, col=0, colspan=3)

fig, axes = g.build(fig_width='3.5in', fig_height='3in')

axes[p_img_left].set_title('Image A')
axes[p_img_right].set_title('Image B')
axes[p_data].set_title('Data')
```

`build()` returns the figure and a `dict` mapping each `Panel` handle to its
`matplotlib.axes.Axes`. The `fig_height` is optional for layouts that contain
no `'fr'` rows — in that case the height is inferred from the aspect
constraints.

## API

### `Grid(rows, cols, *, hgap=0, wgap=0, gap=None, margin=0)`

Create a layout grid.

- **`rows`**, **`cols`** — lists of size specs, one per track.
- **`hgap`**, **`wgap`** — gap between rows / columns. A single spec or a list
  of `nrows−1` / `ncols−1` values for per-gap control. Must be fixed sizes.
- **`gap`** — shorthand that sets both `hgap` and `wgap` to the same value.
- **`margin`** — whitespace around the content area. One value (all sides),
  two values (vertical, horizontal), or four values (top, right, bottom, left).

### `Grid.panel(row, col, *, rowspan=1, colspan=1, aspect=None) → Panel`

Register a panel. Returns an opaque `Panel` handle used as the key in the
`axes` dict returned by `build()`.

- **`aspect`** — height-to-width ratio. Required for `'auto'` tracks; ignored
  for fully fixed/fr tracks.

### `Grid.fill(aspect=None) → list[list[Panel]]`

Create one panel per cell and return them as `panels[row][col]`.

### `Grid.subgrid(row, col, *, rowspan=1, colspan=1, rows, cols, hgap=0, wgap=0, gap=None) → Grid`

Nest a child grid inside the given cell(s). Returns the child `Grid`, which
supports the same `panel()`, `fill()`, and `subgrid()` calls.

### `Grid.build(fig_width, fig_height=None, n_iter=1) → (Figure, dict[Panel, Axes])`

Solve the layout and create the figure. `fig_width` is required and must be a
fixed size. `fig_height` is inferred when omitted (not allowed for `'fr'` rows).

- **`n_iter`** — number of layout passes the engine runs per draw call. The
  default of `1` is sufficient for most layouts. Use `2`–`3` when tick labels
  or other decorations shift the layout enough that a second measurement
  produces noticeably different gaps. The engine stops early once the layout
  has converged.

## Examples

| File | What it shows |
|---|---|
| [examples/basic.ipynb](examples/basic.ipynb) | Two square panels side by side with a gap and margin |
| [examples/mixed_sizing.ipynb](examples/mixed_sizing.ipynb) | Fixed header row, fractional body row, auto-sized image columns |
| [examples/subgrid.ipynb](examples/subgrid.ipynb) | 2×2 image grid nested inside one cell of a larger layout |

---

## Disclamer
This package was built with Claude Code.
