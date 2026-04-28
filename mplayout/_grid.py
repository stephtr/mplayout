"""
Constraint-based matplotlib figure layout.

Size specs
----------
number / '1.5in' / '15mm' / '1cm'   fixed size in the given unit
'1fr' / '2fr'                         proportional share of remaining space
                                       (requires the figure dimension to be given)
'auto'                                determined by aspect-ratio constraints

Quick example
-------------
    from mplayout import Grid

    g = Grid(
        rows=['auto', '1fr'],
        cols=['auto', '5mm', 'auto'],
        hgap='2mm', wgap='2mm', margin='5mm',
    )
    p_img_left  = g.panel(row=0, col=0, aspect=1.0)
    p_img_right = g.panel(row=0, col=2, aspect=4/3)
    p_data      = g.panel(row=1, col=0, colspan=3)

    fig, axes = g.build(fig_width='3.5in')
    axes[p_img_left].imshow(...)
    axes[p_data].plot(...)

Subgrid example
---------------
    g = Grid(rows=['auto', '5mm', 'auto'], cols=['1fr', '3mm', '3mm'],
             margin='5mm')

    sub = g.subgrid(row=0, col=0,
                    rows=['auto', 'auto'], cols=['auto', 'auto'],
                    hgap='1mm', wgap='1mm')
    p00 = sub.panel(row=0, col=0, aspect=1.0)
    p01 = sub.panel(row=0, col=1, aspect=1.0)
    p10 = sub.panel(row=1, col=0, aspect=1.0)
    p11 = sub.panel(row=1, col=1, aspect=1.0)

    p_cb0 = g.panel(row=0, col=2)       # colorbar slot (only for row 0)
    p_big = g.panel(row=2, col=0, colspan=3)

    fig, axes = g.build(fig_width='3.5in')
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.layout_engine import LayoutEngine

SizeSpec = Union[str, float, int]

_UNSET: Any = object()  # sentinel: distinguishes "not passed" from 0

# ─── Unit parsing ──────────────────────────────────────────────────────────────

_UNITS: Dict[str, float] = {'in': 1.0, 'mm': 1 / 25.4, 'cm': 1 / 2.54}


def _parse_size(spec: SizeSpec) -> Tuple[str, Optional[float]]:
    """Return ('fixed', inches), ('fr', weight), or ('auto', None)."""
    if spec == 'auto':
        return 'auto', None
    if isinstance(spec, (int, float)):
        return 'fixed', float(spec)
    if isinstance(spec, str):
        s = spec.strip()
        if s == 'auto':
            return 'auto', None
        m = re.fullmatch(r'(\d+(?:\.\d*)?)\s*(fr|in|mm|cm)', s)
        if m:
            v, u = float(m.group(1)), m.group(2)
            return ('fr', v) if u == 'fr' else ('fixed', v * _UNITS[u])
    raise ValueError(f"Cannot parse size spec: {spec!r}")


def _parse_gaps(spec: Union[SizeSpec, List[SizeSpec]], n: int) -> List[float]:
    """Resolve a gap spec into a list of *n* fixed values (inches)."""
    if n == 0:
        return []
    if spec is None or spec == 0:
        return [0.0] * n
    if isinstance(spec, (list, tuple)):
        if len(spec) != n:
            raise ValueError(f"Expected {n} gap values, got {len(spec)}")
        result = []
        for s in spec:
            kind, val = _parse_size(s)
            if kind != 'fixed':
                raise ValueError(f"Gaps must be fixed sizes, not {s!r}")
            result.append(val)
        return result
    kind, val = _parse_size(spec)
    if kind != 'fixed':
        raise ValueError(f"Gaps must be fixed sizes, not {spec!r}")
    return [val] * n


def _parse_margin(spec) -> Tuple[float, float, float, float]:
    """Return (top, right, bottom, left) in inches."""
    if spec is None or spec == 0:
        return 0.0, 0.0, 0.0, 0.0
    if isinstance(spec, (int, float, str)):
        v = _parse_size(spec)[1]
        return v, v, v, v
    vals: List[float] = []
    for s in spec:
        kind, v = _parse_size(s)
        if kind != 'fixed' or v is None:
            raise ValueError(f"Margin must be a fixed size, not {s!r}")
        vals.append(v)
    if len(vals) == 2:
        return vals[0], vals[1], vals[0], vals[1]   # (vert, horiz)
    if len(vals) == 4:
        return (vals[0], vals[1], vals[2], vals[3])  # (top, right, bottom, left)
    raise ValueError(f"Cannot parse margin: {spec!r}")


# ─── Panel handle ──────────────────────────────────────────────────────────────


class Panel:
    """
    Opaque handle returned by :meth:`Grid.panel`.

    After :meth:`Grid.build` the corresponding :class:`~matplotlib.axes.Axes`
    is available as ``panel.axes`` and as the value in the returned dict.
    """

    def __init__(
        self,
        r0: int, c0: int,
        rs: int, cs: int,
        aspect: Optional[float],
        grid: 'Grid',
    ) -> None:
        self._r0 = r0
        self._c0 = c0
        self._rs = rs
        self._cs = cs
        self._aspect = aspect
        self._grid = grid
        self.axes: Optional[plt.Axes] = None   # set after build()

    def __repr__(self) -> str:
        row_s = f"row={self._r0}" if self._rs == 1 else f"rows={self._r0}:{self._r0+self._rs}"
        col_s = f"col={self._c0}" if self._cs == 1 else f"cols={self._c0}:{self._c0+self._cs}"
        asp = f", aspect={self._aspect}" if self._aspect is not None else ""
        return f"Panel({row_s}, {col_s}{asp})"


# ─── Grid ──────────────────────────────────────────────────────────────────────


class Grid:
    """
    Constraint-based layout grid for matplotlib figures.

    The layout is solved as a linear system before any Axes are created, so
    the figure size and all panel positions are exact.

    Parameters
    ----------
    rows, cols:
        Lists of size specs, one per row / column.
    hgap, wgap:
        Gap between rows / columns.  A single spec applies uniformly; a list
        of ``nrows-1`` / ``ncols-1`` specs gives per-gap control.
        Must be fixed sizes.
    margin:
        Whitespace around the content area.  One value (all sides), two
        values (vertical, horizontal), or four values (top, right, bottom,
        left).
    """

    def __init__(
        self,
        rows: List[SizeSpec],
        cols: List[SizeSpec],
        hgap: Union[SizeSpec, List[SizeSpec]] = _UNSET,
        wgap: Union[SizeSpec, List[SizeSpec]] = _UNSET,
        margin: Union[SizeSpec, tuple] = 0,
        gap: Optional[Union[SizeSpec, List[SizeSpec]]] = None,
    ) -> None:
        if gap is not None:
            if hgap is not _UNSET or wgap is not _UNSET:
                raise ValueError("Cannot specify 'gap' together with 'hgap' or 'wgap'")
            hgap = wgap = gap
        if hgap is _UNSET:
            hgap = 0
        if wgap is _UNSET:
            wgap = 0
        self._rspecs: List[Tuple[str, Optional[float]]] = [_parse_size(r) for r in rows]
        self._cspecs: List[Tuple[str, Optional[float]]] = [_parse_size(c) for c in cols]
        self._hgaps: List[float] = _parse_gaps(hgap, len(rows) - 1)
        self._wgaps: List[float] = _parse_gaps(wgap, len(cols) - 1)
        self._margin: Tuple[float, float, float, float] = _parse_margin(margin)
        self._panels: List[Panel] = []
        # (r0, c0, rowspan, colspan, child_Grid)
        self._children: List[Tuple[int, int, int, int, 'Grid']] = []
        self._occupied: set = set()  # (row, col) cells already claimed

        # Variable indices — filled by _assign_vars():
        self._rv: Optional[List[int]] = None   # row-height variables
        self._cv: Optional[List[int]] = None   # col-width  variables
        self._frh: Optional[int] = None        # auxiliary: 1 fr-row   unit (inches)
        self._frw: Optional[int] = None        # auxiliary: 1 fr-col   unit (inches)

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def nrows(self) -> int:
        return len(self._rspecs)

    @property
    def ncols(self) -> int:
        return len(self._cspecs)

    def panel(
        self,
        row: int,
        col: int,
        rowspan: int = 1,
        colspan: int = 1,
        aspect: Optional[float] = None,
    ) -> Panel:
        """
        Register a panel at (*row*, *col*) and return a handle.

        Parameters
        ----------
        aspect:
            Height-to-width ratio.  Drives the size of 'auto' rows/cols that
            the panel occupies.
        """
        _check_span(row, rowspan, self.nrows, 'row')
        _check_span(col, colspan, self.ncols, 'col')
        for r in range(row, row + rowspan):
            for c in range(col, col + colspan):
                if (r, c) in self._occupied:
                    raise ValueError(f"Cell ({r}, {c}) is already occupied")
        for r in range(row, row + rowspan):
            for c in range(col, col + colspan):
                self._occupied.add((r, c))
        p = Panel(row, col, rowspan, colspan, aspect, self)
        self._panels.append(p)
        return p

    def fill(
        self,
        aspect: Optional[float] = None,
    ) -> List[List[Panel]]:
        """
        Create a panel in every cell and return them as ``panels[row][col]``.

        Equivalent to calling :meth:`panel` for each position in the grid.
        Raises ``ValueError`` if any cell is already occupied.

        Parameters
        ----------
        aspect:
            Height-to-width ratio passed to every panel.  Pass ``None`` for
            panels whose size is determined solely by fixed or 'fr' tracks.
        """
        result: List[List[Panel]] = []
        for r in range(self.nrows):
            row_panels: List[Panel] = []
            for c in range(self.ncols):
                row_panels.append(self.panel(row=r, col=c, aspect=aspect))
            result.append(row_panels)
        return result

    def subgrid(
        self,
        row: int,
        col: int,
        rowspan: int = 1,
        colspan: int = 1,
        rows: Optional[List[SizeSpec]] = None,
        cols: Optional[List[SizeSpec]] = None,
        hgap: Union[SizeSpec, List[SizeSpec]] = _UNSET,
        wgap: Union[SizeSpec, List[SizeSpec]] = _UNSET,
        gap: Optional[Union[SizeSpec, List[SizeSpec]]] = None,
    ) -> 'Grid':
        """
        Register a child grid that fills the given cell(s) and return it.

        The child grid has no margin; its rows and columns fill the parent
        cell exactly (after accounting for gaps on both sides).
        """
        if gap is not None:
            if hgap is not _UNSET or wgap is not _UNSET:
                raise ValueError("Cannot specify 'gap' together with 'hgap' or 'wgap'")
            hgap = wgap = gap
        if hgap is _UNSET:
            hgap = 0
        if wgap is _UNSET:
            wgap = 0
        _check_span(row, rowspan, self.nrows, 'row')
        _check_span(col, colspan, self.ncols, 'col')
        for r in range(row, row + rowspan):
            for c in range(col, col + colspan):
                if (r, c) in self._occupied:
                    raise ValueError(f"Cell ({r}, {c}) is already occupied")
        for r in range(row, row + rowspan):
            for c in range(col, col + colspan):
                self._occupied.add((r, c))
        child = Grid(
            rows=rows or ['auto'],
            cols=cols or ['auto'],
            hgap=hgap,
            wgap=wgap,
            margin=0,
        )
        self._children.append((row, col, rowspan, colspan, child))
        return child

    def build(
        self,
        fig_width: SizeSpec,
        fig_height: Optional[SizeSpec] = None,
        n_iter: int = 1,
    ) -> Tuple[plt.Figure, Dict[Panel, plt.Axes]]:
        """
        Solve the layout and create a matplotlib figure.

        Parameters
        ----------
        fig_width:
            Figure width (required).
        fig_height:
            Figure height.  When *None* (default) the height is derived from
            the row sizes and aspect constraints.  Must be given if any root
            row uses ``'fr'`` sizing.
        n_iter:
            Number of layout passes the engine runs per draw.  One pass is
            usually sufficient; use 2–3 when labels shift the layout enough
            that a second measurement produces noticeably different gaps.
            The engine stops early if the layout has already converged.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : dict mapping each Panel to its Axes
        """
        _fw_kind, fw = _parse_size(fig_width)
        if _fw_kind != 'fixed' or fw is None:
            raise ValueError(f"fig_width must be a fixed size, not {fig_width!r}")
        fh = _parse_size(fig_height)[1] if fig_height is not None else None
        fh_auto = (fig_height is None)

        if fh is None:
            for kind, _ in self._rspecs:
                if kind == 'fr':
                    raise ValueError(
                        "Root grid contains 'fr' rows but fig_height was not given. "
                        "Pass fig_height= to build(), or replace 'fr' rows with 'auto'."
                    )

        # ── Step 1: assign global variable indices ──────────────────────────
        ctr = [0]
        self._assign_vars(ctr)
        n_vars = ctr[0]

        # ── Step 2: build the linear constraint system ──────────────────────
        A_rows: List[np.ndarray] = []
        b_rows: List[float] = []
        self._collect_constraints(A_rows, b_rows, n_vars, fw, fh, is_root=True)

        A = np.array(A_rows, dtype=float)
        b = np.array(b_rows, dtype=float)

        # ── Step 3: solve ───────────────────────────────────────────────────
        if A.shape[0] == n_vars:
            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                raise RuntimeError(
                    "Layout is singular — check for 'auto' tracks with no aspect "
                    "constraint and no other size constraint."
                )
            rank = n_vars
        else:
            x, _, _rank, _ = np.linalg.lstsq(A, b, rcond=None)
            rank = int(_rank)
        _validate_solution(A, b, x, rank, n_vars, rv=self._rv, cv=self._cv)

        # ── Step 4: compute auto figure height ──────────────────────────────
        if fh is None:
            m = self._margin
            fh = sum(x[v] for v in self._rv) + sum(self._hgaps) + m[0] + m[2]

        # ── Step 5: create figure, place axes, install layout engine ────────
        fig = plt.figure(figsize=(fw, fh))
        axes_map: Dict[Panel, plt.Axes] = {}
        m = self._margin
        self._place(fig, x, axes_map, fw, fh,
                    x_left=m[3],
                    y_top=fh - m[0])
        col_left, col_right, row_bot, row_top = self._compute_boundaries(
            x, self._hgaps, self._wgaps, m
        )
        engine = GridLayoutEngine(
            grid=self,
            axes_panels=axes_map,
            fig_width=fw,
            fh_auto=fh_auto,
            user_hgaps=list(self._hgaps),
            user_wgaps=list(self._wgaps),
            user_margin=m,
            n_vars=n_vars,
            col_left=col_left,
            col_right=col_right,
            row_bot=row_bot,
            row_top=row_top,
            n_iter=n_iter,
        )
        fig.set_layout_engine(engine)
        return fig, axes_map

    # ── Variable assignment ────────────────────────────────────────────────────

    def _assign_vars(self, ctr: List[int]) -> None:
        """Depth-first assignment of global variable indices."""
        nr, nc = self.nrows, self.ncols
        self._rv = list(range(ctr[0], ctr[0] + nr));  ctr[0] += nr
        self._cv = list(range(ctr[0], ctr[0] + nc));  ctr[0] += nc
        # Auxiliary variables for fractional sizing
        if any(k == 'fr' for k, _ in self._rspecs):
            self._frh = ctr[0];  ctr[0] += 1
        if any(k == 'fr' for k, _ in self._cspecs):
            self._frw = ctr[0];  ctr[0] += 1
        for *_, child in self._children:
            child._assign_vars(ctr)

    # ── Constraint collection ──────────────────────────────────────────────────

    def _collect_constraints(
        self,
        A: List[np.ndarray],
        b: List[float],
        n_vars: int,
        fw: float,
        fh: Optional[float],
        is_root: bool = False,
    ) -> None:
        rv, cv = self._rv, self._cv

        def add(coeffs: Dict[int, float], rhs: float) -> None:
            row = np.zeros(n_vars)
            for idx, coef in coeffs.items():
                row[idx] += coef
            A.append(row)
            b.append(float(rhs))

        # ── Fixed sizes ──────────────────────────────────────────────────────
        for i, (kind, val) in enumerate(self._rspecs):
            if kind == 'fixed':
                add({rv[i]: 1.0}, val)
        for j, (kind, val) in enumerate(self._cspecs):
            if kind == 'fixed':
                add({cv[j]: 1.0}, val)

        # ── Fractional sizing: x[i] = weight × unit_fr ──────────────────────
        #    The unit_fr variable absorbs the "what is 1fr in inches" question.
        #    Together with the sum constraint below it is fully determined.
        for i, (kind, val) in enumerate(self._rspecs):
            if kind == 'fr':
                add({rv[i]: 1.0, self._frh: -val}, 0.0)
        for j, (kind, val) in enumerate(self._cspecs):
            if kind == 'fr':
                add({cv[j]: 1.0, self._frw: -val}, 0.0)

        # ── Sum of all cols / rows = available space after margins + gaps ────
        #    For the root grid these equations bind the layout to the figure
        #    size. For child grids the equivalent role is played by the
        #    coupling constraints below.
        if is_root:
            m = self._margin
            add(
                {cv[j]: 1.0 for j in range(self.ncols)},
                fw - sum(self._wgaps) - m[1] - m[3],
            )
            if fh is not None:
                add(
                    {rv[i]: 1.0 for i in range(self.nrows)},
                    fh - sum(self._hgaps) - m[0] - m[2],
                )

        # ── Aspect-ratio constraints ─────────────────────────────────────────
        #    sum(h_rows_in_span) - aspect × sum(w_cols_in_span)
        #        = aspect × wgaps_in_span − hgaps_in_span
        for p in self._panels:
            if p._aspect is None:
                continue
            r0, c0 = p._r0, p._c0
            r1, c1 = r0 + p._rs, c0 + p._cs
            hg = sum(self._hgaps[r0:r1 - 1])   # gaps *inside* the row span
            wg = sum(self._wgaps[c0:c1 - 1])   # gaps *inside* the col span
            coeffs: Dict[int, float] = {}
            for i in range(r0, r1):
                coeffs[rv[i]] = coeffs.get(rv[i], 0.0) + 1.0
            for j in range(c0, c1):
                coeffs[cv[j]] = coeffs.get(cv[j], 0.0) - p._aspect
            add(coeffs, p._aspect * wg - hg)

        # ── Subgrid coupling + recursion ─────────────────────────────────────
        for (r0, c0, rs, cs, child) in self._children:
            # Collect child's own constraints first
            child._collect_constraints(A, b, n_vars, fw, fh, is_root=False)

            # Width: sum(child cols) + child internal wgaps
            #      = sum(parent cols in span) + parent internal wgaps
            child_wg  = sum(child._wgaps)
            parent_wg = sum(self._wgaps[c0:c0 + cs - 1])
            coeffs = {child._cv[j]: 1.0 for j in range(child.ncols)}
            for j in range(c0, c0 + cs):
                coeffs[cv[j]] = coeffs.get(cv[j], 0.0) - 1.0
            add(coeffs, parent_wg - child_wg)

            # Height: sum(child rows) + child internal hgaps
            #       = sum(parent rows in span) + parent internal hgaps
            child_hg  = sum(child._hgaps)
            parent_hg = sum(self._hgaps[r0:r0 + rs - 1])
            coeffs = {child._rv[i]: 1.0 for i in range(child.nrows)}
            for i in range(r0, r0 + rs):
                coeffs[rv[i]] = coeffs.get(rv[i], 0.0) - 1.0
            add(coeffs, parent_hg - child_hg)

    # ── Axes placement ─────────────────────────────────────────────────────────

    def _place(
        self,
        fig: plt.Figure,
        x: np.ndarray,
        axes_map: Dict[Panel, plt.Axes],
        fw: float,
        fh: float,
        x_left: float,   # left edge of this grid's content area (inches from fig left)
        y_top: float,    # top  edge of this grid's content area (inches from fig bottom)
    ) -> None:
        cw = [x[v] for v in self._cv]
        rh = [x[v] for v in self._rv]

        # Left edge of each column (inches from figure left)
        col_x = np.empty(self.ncols)
        col_x[0] = x_left
        for j in range(1, self.ncols):
            col_x[j] = col_x[j - 1] + cw[j - 1] + self._wgaps[j - 1]

        # Bottom edge of each row (inches from figure bottom).
        # Row 0 is topmost, so y decreases with increasing row index.
        row_y = np.empty(self.nrows)
        row_y[0] = y_top - rh[0]
        for i in range(1, self.nrows):
            row_y[i] = row_y[i - 1] - self._hgaps[i - 1] - rh[i]

        # Create an Axes for each panel
        for p in self._panels:
            r0, c0 = p._r0, p._c0
            r1, c1 = r0 + p._rs, c0 + p._cs
            px = col_x[c0] / fw
            py = row_y[r1 - 1] / fh                                   # bottom of span
            pw = (sum(cw[c0:c1]) + sum(self._wgaps[c0:c1 - 1])) / fw
            ph = (sum(rh[r0:r1]) + sum(self._hgaps[r0:r1 - 1])) / fh
            ax = fig.add_axes([px, py, pw, ph])
            p.axes = ax
            axes_map[p] = ax

        # Recurse into subgrids
        for (r0, c0, rs, cs, child) in self._children:
            child._place(
                fig, x, axes_map, fw, fh,
                x_left=col_x[c0],
                y_top=row_y[r0] + rh[r0],    # top of the topmost row in the span
            )

    # ── Layout-engine helpers ──────────────────────────────────────────────────

    def _compute_boundaries(
        self,
        x: np.ndarray,
        hgaps: List[float],
        wgaps: List[float],
        margin: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (col_left, col_right, row_bot, row_top) in inches from fig bottom-left."""
        cw = [x[v] for v in self._cv]
        rh = [x[v] for v in self._rv]
        nc, nr = self.ncols, self.nrows

        col_left  = np.empty(nc)
        col_right = np.empty(nc)
        col_left[0]  = margin[3]
        col_right[0] = col_left[0] + cw[0]
        for j in range(1, nc):
            col_left[j]  = col_right[j - 1] + wgaps[j - 1]
            col_right[j] = col_left[j] + cw[j]

        row_bot = np.empty(nr)
        row_top = np.empty(nr)
        row_bot[nr - 1] = margin[2]
        row_top[nr - 1] = row_bot[nr - 1] + rh[nr - 1]
        for i in range(nr - 2, -1, -1):
            row_bot[i] = row_top[i + 1] + hgaps[i]
            row_top[i] = row_bot[i] + rh[i]

        return col_left, col_right, row_bot, row_top

    def _compute_positions(
        self,
        x: np.ndarray,
        hgaps: List[float],
        wgaps: List[float],
        fw: float,
        fh: float,
        x_left: float,
        y_top: float,
    ) -> Dict['Panel', Tuple[float, float, float, float]]:
        """Return Panel → (x, y, w, h) (figure fractions) without creating Axes."""
        cw = [x[v] for v in self._cv]
        rh = [x[v] for v in self._rv]

        col_x = np.empty(self.ncols)
        col_x[0] = x_left
        for j in range(1, self.ncols):
            col_x[j] = col_x[j - 1] + cw[j - 1] + wgaps[j - 1]

        row_y = np.empty(self.nrows)
        row_y[0] = y_top - rh[0]
        for i in range(1, self.nrows):
            row_y[i] = row_y[i - 1] - hgaps[i - 1] - rh[i]

        positions: Dict['Panel', Tuple[float, float, float, float]] = {}
        for p in self._panels:
            r0, c0 = p._r0, p._c0
            r1, c1 = r0 + p._rs, c0 + p._cs
            positions[p] = (
                float(col_x[c0] / fw),
                float(row_y[r1 - 1] / fh),
                float((sum(cw[c0:c1]) + sum(wgaps[c0:c1 - 1])) / fw),
                float((sum(rh[r0:r1]) + sum(hgaps[r0:r1 - 1])) / fh),
            )

        for (r0, c0, rs, cs, child) in self._children:
            positions.update(child._compute_positions(
                x, child._hgaps, child._wgaps, fw, fh,
                x_left=col_x[c0],
                y_top=row_y[r0] + rh[r0],
            ))

        return positions

    def _resolve(
        self,
        fig_width: float,
        fig_height: Optional[float],
        hgaps: List[float],
        wgaps: List[float],
        margin: Tuple[float, float, float, float],
        n_vars: int,
    ) -> Tuple[np.ndarray, float]:
        """Re-solve the layout with new gap/margin values (all pre-parsed, in inches)."""
        old_hgaps, old_wgaps, old_margin = self._hgaps, self._wgaps, self._margin
        self._hgaps = hgaps
        self._wgaps = wgaps
        self._margin = margin
        try:
            A_rows: List[np.ndarray] = []
            b_rows: List[float] = []
            self._collect_constraints(A_rows, b_rows, n_vars, fig_width, fig_height,
                                      is_root=True)
            A = np.array(A_rows, dtype=float)
            b = np.array(b_rows, dtype=float)
            if A.shape[0] == n_vars:
                try:
                    x = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    raise RuntimeError(
                        "Layout is singular — check for 'auto' tracks with no aspect "
                        "constraint and no other size constraint."
                    )
                rank = n_vars
            else:
                x, _, _rank, _ = np.linalg.lstsq(A, b, rcond=None)
                rank = int(_rank)
            _validate_solution(A, b, x, rank, n_vars, rv=self._rv, cv=self._cv)
            fh = fig_height
            if fh is None:
                assert self._rv is not None  # always set after build()
                fh = sum(x[v] for v in self._rv) + sum(hgaps) + margin[0] + margin[2]
            return x, fh
        finally:
            self._hgaps = old_hgaps
            self._wgaps = old_wgaps
            self._margin = old_margin


# ─── Helpers ───────────────────────────────────────────────────────────────────


def _check_span(start: int, span: int, n: int, axis: str) -> None:
    if not (0 <= start < n):
        raise ValueError(f"{axis} index {start} out of range [0, {n})")
    if span < 1 or start + span > n:
        raise ValueError(
            f"{axis} span of {span} starting at {start} exceeds grid size {n}"
        )


def _validate_solution(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    rank: int,
    n_vars: int,
    rv: Optional[List[int]] = None,
    cv: Optional[List[int]] = None,
) -> None:
    if rank < n_vars:
        raise RuntimeError(
            f"Layout is underdetermined: {n_vars} unknown sizes but only "
            f"{rank} independent constraints.  "
            "Add aspect ratios or replace 'auto' specs with fixed sizes."
        )
    per_row = np.abs(A @ x - b)
    residual = float(per_row.max())
    if residual > 1e-8:
        bad_var_set = set()
        for ri in np.where(per_row > 1e-8)[0]:
            bad_var_set.update(int(i) for i in np.where(np.abs(A[ri]) > 0)[0])
        parts: List[str] = []
        if rv is not None:
            for track_i, vi in enumerate(rv):
                if vi in bad_var_set:
                    parts.append(f"  row {track_i} height (aspect constraints from multiple panels require different values)")
        if cv is not None:
            for track_j, vj in enumerate(cv):
                if vj in bad_var_set:
                    parts.append(f"  col {track_j} width (aspect constraints from multiple panels require different values)")
        detail = ("\nConflicting constraints involve:\n" + "\n".join(parts)) if parts else ""
        raise RuntimeError(
            f"Layout constraints are inconsistent (max residual = {residual:.2e}).{detail}\n"
            "Check panels with aspect ratios that share an 'auto' row or column."
        )
    if np.any(x < -1e-9):
        raise RuntimeError(
            f"Solver produced a negative size (min = {float(x.min()):.4g} in).  "
            "The constraints may be infeasible — check that margins + gaps "
            "do not exceed the figure width."
        )


# ─── Layout engine ─────────────────────────────────────────────────────────────


class GridLayoutEngine(LayoutEngine):
    """
    Installed by :meth:`Grid.build`; called by matplotlib before every draw.

    Measures axis-label overhangs, expands gaps/margins to accommodate them,
    re-solves the constraint system, and repositions all Axes.  The figure
    width is always conserved; the height grows only when it was auto-computed.
    """

    def __init__(
        self,
        grid: Grid,
        axes_panels: Dict[Panel, Any],
        fig_width: float,
        fh_auto: bool,
        user_hgaps: List[float],
        user_wgaps: List[float],
        user_margin: Tuple[float, float, float, float],
        n_vars: int,
        col_left: np.ndarray,
        col_right: np.ndarray,
        row_bot: np.ndarray,
        row_top: np.ndarray,
        n_iter: int = 1,
    ) -> None:
        super().__init__()
        self._adjust_compatible = False
        self._colorbar_gridspec = False
        self._grid      = grid
        self._axes_panels = axes_panels
        self._fw        = fig_width
        self._fh_auto   = fh_auto
        self._user_hgaps = user_hgaps
        self._user_wgaps = user_wgaps
        self._user_margin = user_margin
        self._n_vars    = n_vars
        self._col_left  = col_left
        self._col_right = col_right
        self._row_bot   = row_bot
        self._row_top   = row_top
        self._n_iter    = n_iter
        # Cache
        self._last_fig_size: Optional[Tuple[float, float]] = None
        self._last_eff_hgaps: Optional[Tuple[float, ...]] = None
        self._last_eff_wgaps: Optional[Tuple[float, ...]] = None
        self._last_eff_margin: Optional[Tuple[float, float, float, float]] = None

    def get(self) -> Dict[str, Any]:
        return {}

    def execute(self, fig: Any) -> None:
        try:
            renderer = fig._get_renderer()  # type: ignore[attr-defined]
        except AttributeError:
            return  # backend has no renderer; skip
        for _ in range(self._n_iter):
            if not self._step(fig, renderer):
                break  # layout converged; no need for further passes

    def _step(self, fig: Any, renderer: Any) -> bool:
        """Run one layout pass. Returns True if a re-solve happened."""
        dpi = fig.dpi
        fig_w, fig_h = fig.get_size_inches()
        tol = 1e-3  # inches

        nr = self._grid.nrows
        nc = self._grid.ncols
        col_left  = self._col_left
        col_right = self._col_right
        row_bot   = self._row_bot
        row_top   = self._row_top

        # Per-slot overhang collectors
        left_margin_oh:  List[float] = []
        right_margin_oh: List[float] = []
        top_margin_oh:   List[float] = []
        bot_margin_oh:   List[float] = []
        right_into_gap: List[List[float]] = [[] for _ in range(nc - 1)]
        left_into_gap:  List[List[float]] = [[] for _ in range(nc - 1)]
        bot_into_gap:   List[List[float]] = [[] for _ in range(nr - 1)]
        top_into_gap:   List[List[float]] = [[] for _ in range(nr - 1)]

        for ax in fig.axes:
            tight = ax.get_tightbbox(renderer)
            if tight is None:
                continue
            frame = ax.get_window_extent(renderer)

            oh_left   = max(0.0, (frame.x0 - tight.x0) / dpi)
            oh_right  = max(0.0, (tight.x1 - frame.x1) / dpi)
            oh_bottom = max(0.0, (frame.y0 - tight.y0) / dpi)
            oh_top    = max(0.0, (tight.y1 - frame.y1) / dpi)

            # Outer margins: measure how far the tight bbox extends past the
            # content boundary, for every axis regardless of which track it is in.
            left_margin_oh.append(max(0.0,  col_left[0]     - tight.x0 / dpi))
            right_margin_oh.append(max(0.0, tight.x1 / dpi  - col_right[nc - 1]))
            bot_margin_oh.append(max(0.0,   row_bot[nr - 1] - tight.y0 / dpi))
            top_margin_oh.append(max(0.0,   tight.y1 / dpi  - row_top[0]))

            pos      = ax.get_position()
            ax_left  = pos.x0 * fig_w
            ax_right = (pos.x0 + pos.width)  * fig_w
            ax_bot   = pos.y0 * fig_h
            ax_top   = (pos.y0 + pos.height) * fig_h

            # Column-gap slots
            for j in range(nc - 1):
                if abs(ax_right - col_right[j])    < tol: right_into_gap[j].append(oh_right)
                if abs(ax_left  - col_left[j + 1]) < tol: left_into_gap[j].append(oh_left)

            # Row-gap slots (row 0 is top; row_bot[i] > row_top[i+1])
            for i in range(nr - 1):
                if abs(ax_bot - row_bot[i])      < tol: bot_into_gap[i].append(oh_bottom)
                if abs(ax_top - row_top[i + 1])  < tol: top_into_gap[i].append(oh_top)

        # Aggregate needs per slot
        wgap_needs = [
            max(right_into_gap[j], default=0.0) + max(left_into_gap[j], default=0.0)
            for j in range(nc - 1)
        ]
        hgap_needs = [
            max(bot_into_gap[i], default=0.0) + max(top_into_gap[i], default=0.0)
            for i in range(nr - 1)
        ]
        eff_wgaps  = [max(u, m) for u, m in zip(self._user_wgaps, wgap_needs)]
        eff_hgaps  = [max(u, m) for u, m in zip(self._user_hgaps, hgap_needs)]
        eff_margin = (
            max(self._user_margin[0], max(top_margin_oh,  default=0.0)),
            max(self._user_margin[1], max(right_margin_oh, default=0.0)),
            max(self._user_margin[2], max(bot_margin_oh,  default=0.0)),
            max(self._user_margin[3], max(left_margin_oh, default=0.0)),
        )

        # Cache check — skip re-solve if nothing changed
        eff_hgaps_t  = tuple(eff_hgaps)
        eff_wgaps_t  = tuple(eff_wgaps)
        current_size = (float(fig_w), float(fig_h))
        if (current_size       == self._last_fig_size
                and eff_hgaps_t  == self._last_eff_hgaps
                and eff_wgaps_t  == self._last_eff_wgaps
                and eff_margin   == self._last_eff_margin):
            return False

        # Re-solve with effective gaps/margins
        current_fh = None if self._fh_auto else fig_h
        x, new_fh = self._grid._resolve(
            fig_width=self._fw,
            fig_height=current_fh,
            hgaps=eff_hgaps,
            wgaps=eff_wgaps,
            margin=eff_margin,
            n_vars=self._n_vars,
        )

        # Reposition every Axes from the new solution
        m = eff_margin
        positions = self._grid._compute_positions(
            x, eff_hgaps, eff_wgaps, self._fw, new_fh,
            x_left=m[3], y_top=new_fh - m[0],
        )
        for panel, ax in self._axes_panels.items():
            if panel in positions:
                ax.set_position(positions[panel])

        # Grow figure height if it was auto-computed
        if self._fh_auto:
            fig.set_size_inches(self._fw, new_fh)

        # Update boundary arrays for the next _step() call
        self._col_left, self._col_right, self._row_bot, self._row_top = (
            self._grid._compute_boundaries(x, eff_hgaps, eff_wgaps, eff_margin)
        )

        # Store cache state (after potential set_size_inches)
        sz = fig.get_size_inches()
        self._last_fig_size  = (float(sz[0]), float(sz[1]))
        self._last_eff_hgaps = eff_hgaps_t
        self._last_eff_wgaps = eff_wgaps_t
        self._last_eff_margin = eff_margin
        return True