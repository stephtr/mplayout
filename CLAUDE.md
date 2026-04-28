# grid.py — Constraint-based matplotlib layout

## What this is

A Python modulethat provides a flexible, constraint-based
figure layout engine for matplotlib. It replaces `GridSpec` and `constrained_layout`
for cases that require mixed sizing units, aspect-ratio-driven track sizing, and
exact colorbar alignment across rows.

## Core concept

Every row height and column width is a variable. The layout is expressed as a
linear system Ax = b and solved once before any Axes are created. The figure size
and all panel positions are therefore exact — no post-hoc adjustment.

## Sizing vocabulary

| Spec | Meaning |
|---|---|
| `1.5`, `"1.5in"`, `"15mm"`, `"1cm"` | Fixed size in the given unit |
| `"1fr"`, `"2fr"` | Proportional share of remaining space (requires the figure dimension to be known) |
| `"auto"` | Determined by aspect-ratio constraints from panels in that track |

## Public API

```python
# Create a grid
g = Grid(
    rows=["auto", "1fr"],
    cols=["auto", "5mm", "auto"],
    hgap="2mm",    # uniform or list of nrows-1 values
    wgap="2mm",
    margin="5mm",  # scalar | (vert, horiz) | (top, right, bottom, left)
)

# Register panels — returns an opaque Panel handle
p = g.panel(row=0, col=0, rowspan=1, colspan=1, aspect=1.0)

# Nest a child grid inside a cell
sub = g.subgrid(row=1, col=2, rows=["1fr","1fr"], cols=["1fr","1fr"], hgap="1mm")
p2  = sub.panel(row=0, col=0, aspect=1.0)

# Solve and build
fig, axes = g.build(fig_width="3.5in")          # fig_height auto-computed
fig, axes = g.build(fig_width="3.5in", fig_height="5in")

# After build, axes are accessible via the panel handle
axes[p].imshow(...)
p.axes.plot(...)   # equivalent
```

## Architecture

### Variable assignment (`_assign_vars`)
Depth-first traversal assigns a contiguous integer index to each row-height and
column-width variable across the root grid and all subgrids. Two auxiliary
variables (`frh`, `frw`) are added per grid that uses `fr` tracks; they represent
"1 fr in inches" for rows and columns respectively.

### Constraint collection (`_collect_constraints`)
Four kinds of linear equations are emitted into the global (A, b) system:

1. **Fixed**: `x[rv[i]] = value`
2. **Fractional**: `x[rv[i]] = weight × frh` (linear ratio constraint) plus a
   sum constraint binding the total of all fr tracks to the remaining space
3. **Aspect ratio**: `Σ h_rows − aspect × Σ w_cols = aspect×wgaps_inside − hgaps_inside`
4. **Subgrid coupling**: child content width/height = parent spanned width/height
   (accounting for gap asymmetry between the two levels)

The root grid additionally gets sum-of-cols and (if `fig_height` is given)
sum-of-rows constraints that anchor the layout to the figure dimensions.

### Solve (`build`)
`numpy.linalg.lstsq` is used with post-solve validation:
- rank check: system must be exactly determined
- residual check: max |Ax − b| must be < 1e-8 in
- negativity check: no variable may be negative

### Placement (`_place`)
After solving, panel positions are computed in inches and converted to matplotlib's
`[left, bottom, width, height]` in figure-fraction coordinates. Subgrids recurse
with the top-left corner of their parent cell as the origin.

## Known missing feature

`Grid.colorbar()` is not yet implemented. The intended design is: a `colorbar()`
call inserts a thin fixed-width column (or row) adjacent to the target panel's
cell, adds a coupling constraint that ties its height (or width) to the panel's
height (or width), and returns a Panel whose Axes can be used as a colorbar host.
This is a first-class grid citizen so that colorbar columns participate in
column-width accounting and preserve plot-edge alignment across rows.