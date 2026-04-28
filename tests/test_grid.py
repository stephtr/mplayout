"""Tests for grid.py layout engine."""

import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mplayout import Grid, Panel
from mplayout._grid import _parse_gaps, _parse_margin


# ── Fix 1: gap list kind validation ───────────────────────────────────────────

class TestGapKindValidation:
    def test_fr_in_gap_list_raises(self):
        with pytest.raises(ValueError, match="Gaps must be fixed sizes"):
            _parse_gaps(['1fr', '2mm'], n=2)

    def test_auto_in_gap_list_raises(self):
        with pytest.raises(ValueError, match="Gaps must be fixed sizes"):
            _parse_gaps(['auto', '2mm'], n=2)

    def test_valid_gap_list_accepted(self):
        gaps = _parse_gaps(['2mm', '3mm'], n=2)
        assert len(gaps) == 2
        assert abs(gaps[0] - 2 / 25.4) < 1e-12
        assert abs(gaps[1] - 3 / 25.4) < 1e-12

    def test_fr_gap_via_grid_constructor_raises(self):
        # Ensure the error surfaces when passed through Grid.__init__
        with pytest.raises(ValueError, match="Gaps must be fixed sizes"):
            Grid(rows=['auto', 'auto', 'auto'], cols=['auto'],
                 hgap=['1fr', '2mm'])


# ── Fix 2: overlapping cell detection ─────────────────────────────────────────

class TestOverlapDetection:
    def test_duplicate_panel_at_same_cell_raises(self):
        g = Grid(rows=['auto', 'auto'], cols=['auto', 'auto'])
        g.panel(row=0, col=0, aspect=1.0)
        with pytest.raises(ValueError, match=r"Cell \(0, 0\) is already occupied"):
            g.panel(row=0, col=0, aspect=2.0)

    def test_spanning_panel_blocks_inner_cells(self):
        g = Grid(rows=['auto', 'auto'], cols=['auto', 'auto'])
        g.panel(row=0, col=0, rowspan=2, colspan=2, aspect=1.0)
        with pytest.raises(ValueError, match=r"Cell \(1, 1\) is already occupied"):
            g.panel(row=1, col=1, aspect=1.0)

    def test_partially_overlapping_spans_raise(self):
        g = Grid(rows=['auto'], cols=['auto', 'auto', 'auto'])
        g.panel(row=0, col=0, colspan=2, aspect=1.0)
        with pytest.raises(ValueError, match=r"Cell \(0, 1\) is already occupied"):
            g.panel(row=0, col=1, colspan=2, aspect=1.0)

    def test_panel_then_subgrid_overlap_raises(self):
        g = Grid(rows=['auto', 'auto'], cols=['auto', 'auto'])
        g.panel(row=0, col=0, aspect=1.0)
        with pytest.raises(ValueError, match=r"Cell \(0, 0\) is already occupied"):
            g.subgrid(row=0, col=0, rows=['auto'], cols=['auto'])

    def test_subgrid_then_panel_overlap_raises(self):
        g = Grid(rows=['auto', 'auto'], cols=['auto', 'auto'])
        g.subgrid(row=1, col=1, rows=['auto'], cols=['auto'])
        with pytest.raises(ValueError, match=r"Cell \(1, 1\) is already occupied"):
            g.panel(row=1, col=1, aspect=1.0)

    def test_non_overlapping_panels_accepted(self):
        g = Grid(rows=['auto'], cols=['auto', 'auto'])
        g.panel(row=0, col=0, aspect=1.0)
        g.panel(row=0, col=1, aspect=1.0)  # must not raise


# ── Fix 3: inconsistent layout error names affected tracks ────────────────────

class TestInconsistentErrorMessage:
    def test_conflicting_aspects_mentions_row(self):
        # Two panels in the same auto row, each with a fixed-width column but
        # incompatible aspect ratios → h_row0 cannot satisfy both constraints.
        g = Grid(rows=['auto'], cols=['1in', '1in'], margin=0)
        g.panel(row=0, col=0, aspect=1.0)   # needs row height = 1 in
        g.panel(row=0, col=1, aspect=2.0)   # needs row height = 2 in → conflict
        with pytest.raises(RuntimeError, match="row 0"):
            g.build(fig_width='2in')

    def test_error_mentions_inconsistent(self):
        g = Grid(rows=['auto'], cols=['1in', '1in'], margin=0)
        g.panel(row=0, col=0, aspect=1.0)
        g.panel(row=0, col=1, aspect=2.0)
        with pytest.raises(RuntimeError, match="inconsistent"):
            g.build(fig_width='2in')


# ── Fix 4: square system uses np.linalg.solve ─────────────────────────────────

class TestSolvePreference:
    def test_square_system_produces_correct_layout(self):
        # rows=['auto'], cols=['auto'], 1 panel aspect=1, no margin.
        # System: 2 equations (sum-of-cols + aspect), 2 unknowns (h0, w0) → square.
        # With fw=2in → w0=2, h0=2 → fig height=2in, axes fills figure.
        g = Grid(rows=['auto'], cols=['auto'], margin=0)
        p = g.panel(row=0, col=0, aspect=1.0)
        fig, axes = g.build(fig_width='2in')
        plt.close(fig)
        bb = axes[p].get_position()
        assert abs(bb.x0) < 1e-9
        assert abs(bb.y0) < 1e-9
        assert abs(bb.width  - 1.0) < 1e-9
        assert abs(bb.height - 1.0) < 1e-9

    def test_auto_row_no_aspect_is_underdetermined(self):
        # An 'auto' row with no aspect constraint cannot be solved.
        g = Grid(rows=['auto'], cols=['auto'], margin=0)
        # No panel → no aspect constraint → underdetermined
        with pytest.raises(RuntimeError):
            g.build(fig_width='2in')


# ── Fix 5: _parse_margin 4-value unpack ───────────────────────────────────────

class TestParseMargin4Values:
    def test_four_values_return_correct_floats(self):
        top, right, bottom, left = _parse_margin(['5mm', '10mm', '3mm', '7mm'])
        assert abs(top    -  5 / 25.4) < 1e-12
        assert abs(right  - 10 / 25.4) < 1e-12
        assert abs(bottom -  3 / 25.4) < 1e-12
        assert abs(left   -  7 / 25.4) < 1e-12

    def test_four_values_are_plain_floats_not_optional(self):
        result = _parse_margin(['5mm', '10mm', '3mm', '7mm'])
        assert isinstance(result, tuple) and len(result) == 4
        assert all(isinstance(v, float) for v in result)


# ── Smoke tests: bounding boxes ───────────────────────────────────────────────

class TestSmokeBoundingBoxes:
    def test_single_panel_with_margin(self):
        # 4in wide, 0.5in margin all sides, auto row+col, aspect=1.
        # Content = 3in × 3in → fig height = 4in.
        # Axes in figure fraction: x0=0.125, y0=0.125, w=0.75, h=0.75.
        g = Grid(rows=['auto'], cols=['auto'], margin='0.5in')
        p = g.panel(row=0, col=0, aspect=1.0)
        fig, axes = g.build(fig_width='4in')
        plt.close(fig)
        bb = axes[p].get_position()
        assert abs(bb.x0     - 0.125) < 1e-6
        assert abs(bb.y0     - 0.125) < 1e-6
        assert abs(bb.width  - 0.75)  < 1e-6
        assert abs(bb.height - 0.75)  < 1e-6

    def test_two_equal_columns_no_gap(self):
        # 4in wide, no margin/gap, two equal auto cols each with aspect=1.
        # Each col = 2in, row height = 2in → fig height = 2in.
        # Left:  x0=0.0, y0=0.0, w=0.5, h=1.0
        # Right: x0=0.5, y0=0.0, w=0.5, h=1.0
        g = Grid(rows=['auto'], cols=['auto', 'auto'], margin=0)
        pl = g.panel(row=0, col=0, aspect=1.0)
        pr = g.panel(row=0, col=1, aspect=1.0)
        fig, axes = g.build(fig_width='4in')
        plt.close(fig)
        lb = axes[pl].get_position()
        rb = axes[pr].get_position()
        assert abs(lb.x0     - 0.0) < 1e-9
        assert abs(lb.y0     - 0.0) < 1e-9
        assert abs(lb.width  - 0.5) < 1e-9
        assert abs(lb.height - 1.0) < 1e-9
        assert abs(rb.x0     - 0.5) < 1e-9
        assert abs(rb.y0     - 0.0) < 1e-9
        assert abs(rb.width  - 0.5) < 1e-9
        assert abs(rb.height - 1.0) < 1e-9

    def test_fixed_height_fr_row(self):
        # Two rows: fixed 1in top, 1fr bottom. Two panels stacked.
        # fig_width=2in, fig_height=3in → bottom row = 3 - 1 = 2in.
        # Top panel:    x0=0, y0=2/3, w=1, h=1/3
        # Bottom panel: x0=0, y0=0,   w=1, h=2/3
        g = Grid(rows=['1in', '1fr'], cols=['auto'], margin=0)
        pt = g.panel(row=0, col=0)
        pb = g.panel(row=1, col=0)
        fig, axes = g.build(fig_width='2in', fig_height='3in')
        plt.close(fig)
        tb = axes[pt].get_position()
        bb = axes[pb].get_position()
        assert abs(tb.y0     - 2/3) < 1e-6
        assert abs(tb.height - 1/3) < 1e-6
        assert abs(bb.y0     - 0.0) < 1e-6
        assert abs(bb.height - 2/3) < 1e-6


# ── gap convenience argument ───────────────────────────────────────────────────

class TestGapArgument:
    def test_gap_sets_both_hgap_and_wgap(self):
        g = Grid(rows=['auto', 'auto'], cols=['auto', 'auto'], gap='4mm')
        expected = 4 / 25.4
        assert abs(g._hgaps[0] - expected) < 1e-12
        assert abs(g._wgaps[0] - expected) < 1e-12

    def test_gap_with_hgap_raises(self):
        with pytest.raises(ValueError, match="Cannot specify 'gap'"):
            Grid(rows=['auto', 'auto'], cols=['auto'], gap='2mm', hgap='3mm')

    def test_gap_with_wgap_raises(self):
        with pytest.raises(ValueError, match="Cannot specify 'gap'"):
            Grid(rows=['auto'], cols=['auto', 'auto'], gap='2mm', wgap='3mm')

    def test_gap_zero_still_works(self):
        g = Grid(rows=['auto', 'auto'], cols=['auto', 'auto'], gap=0)
        assert g._hgaps == [0.0]
        assert g._wgaps == [0.0]

    def test_gap_affects_layout(self):
        # 4in wide, two equal auto cols, gap='4mm', no margin, aspect=1.
        # Each col = (4 - 4/25.4) / 2 in; row height matches col width.
        gap_in = 4 / 25.4
        col_w = (4 - gap_in) / 2
        g = Grid(rows=['auto'], cols=['auto', 'auto'], gap='4mm', margin=0)
        pl = g.panel(row=0, col=0, aspect=1.0)
        pr = g.panel(row=0, col=1, aspect=1.0)
        fig, axes = g.build(fig_width='4in')
        plt.close(fig)
        lb = axes[pl].get_position()
        rb = axes[pr].get_position()
        assert abs(lb.x0)               < 1e-6
        assert abs(lb.width  - col_w/4) < 1e-6
        assert abs(rb.x0 - (col_w + gap_in) / 4) < 1e-6
        assert abs(rb.width  - col_w/4) < 1e-6

    def test_subgrid_gap_sets_both(self):
        g = Grid(rows=['auto'], cols=['auto'])
        sub = g.subgrid(row=0, col=0,
                        rows=['auto', 'auto'], cols=['auto', 'auto'],
                        gap='3mm')
        expected = 3 / 25.4
        assert abs(sub._hgaps[0] - expected) < 1e-12
        assert abs(sub._wgaps[0] - expected) < 1e-12

    def test_subgrid_gap_with_hgap_raises(self):
        g = Grid(rows=['auto'], cols=['auto'])
        with pytest.raises(ValueError, match="Cannot specify 'gap'"):
            g.subgrid(row=0, col=0,
                      rows=['auto', 'auto'], cols=['auto'],
                      gap='2mm', hgap='1mm')


# ── Grid.fill ──────────────────────────────────────────────────────────────────

class TestFill:
    def test_fill_returns_correct_shape(self):
        g = Grid(rows=['auto', 'auto'], cols=['auto', 'auto', 'auto'])
        panels = g.fill(aspect=1.0)
        assert len(panels) == 2
        assert all(len(row) == 3 for row in panels)

    def test_fill_returns_panel_instances(self):
        g = Grid(rows=['auto'], cols=['auto', 'auto'])
        panels = g.fill(aspect=1.0)
        assert all(isinstance(p, Panel) for p in panels[0])

    def test_fill_panels_indexed_correctly(self):
        g = Grid(rows=['auto', 'auto'], cols=['auto', 'auto'])
        panels = g.fill(aspect=1.0)
        assert panels[0][0]._r0 == 0 and panels[0][0]._c0 == 0
        assert panels[0][1]._r0 == 0 and panels[0][1]._c0 == 1
        assert panels[1][0]._r0 == 1 and panels[1][0]._c0 == 0
        assert panels[1][1]._r0 == 1 and panels[1][1]._c0 == 1

    def test_fill_then_panel_raises(self):
        g = Grid(rows=['auto', 'auto'], cols=['auto', 'auto'])
        g.fill(aspect=1.0)
        with pytest.raises(ValueError, match="already occupied"):
            g.panel(row=0, col=0, aspect=1.0)

    def test_fill_twice_raises(self):
        g = Grid(rows=['auto'], cols=['auto'])
        g.fill(aspect=1.0)
        with pytest.raises(ValueError, match="already occupied"):
            g.fill(aspect=2.0)

    def test_fill_builds_and_axes_accessible(self):
        g = Grid(rows=['auto', 'auto'], cols=['auto', 'auto'], gap='2mm', margin='3mm')
        panels = g.fill(aspect=1.0)
        fig, axes = g.build(fig_width='4in')
        plt.close(fig)
        for row in panels:
            for p in row:
                assert p in axes
                assert p.axes is not None

    def test_fill_on_subgrid(self):
        g = Grid(rows=['auto'], cols=['auto'])
        sub = g.subgrid(row=0, col=0, rows=['auto', 'auto'], cols=['auto', 'auto'], gap='1mm')
        panels = sub.fill(aspect=1.0)
        fig, axes = g.build(fig_width='3in')
        plt.close(fig)
        assert len(panels) == 2 and len(panels[0]) == 2
        assert all(p in axes for row in panels for p in row)

    def test_fill_without_aspect(self):
        # fill() with aspect=None on fixed-size tracks should still work
        g = Grid(rows=['1in', '1in'], cols=['1in', '1in'])
        panels = g.fill()
        fig, _ = g.build(fig_width='2in', fig_height='2in')
        plt.close(fig)
        assert len(panels) == 2 and len(panels[0]) == 2


# ── GridLayoutEngine ───────────────────────────────────────────────────────────

def _engine(fig):
    """Return the GridLayoutEngine installed on fig."""
    return fig.get_layout_engine()


class TestGridLayoutEngine:
    def test_no_data_smoke(self):
        # execute() on a freshly built figure with no data must not raise,
        # and all Axes positions must stay in [0, 1].
        g = Grid(rows=['auto'], cols=['auto'], margin='5mm')
        p = g.panel(row=0, col=0, aspect=1.0)
        fig, axes = g.build(fig_width='4in')
        _engine(fig).execute(fig)
        bb = axes[p].get_position()
        assert 0 <= bb.x0 <= 1 and 0 <= bb.y0 <= 1
        assert 0 < bb.width <= 1 and 0 < bb.height <= 1
        plt.close(fig)

    def test_fig_width_conserved_after_execute(self):
        # No matter what labels are present, the figure width must equal fw.
        g = Grid(rows=['auto'], cols=['auto'], margin='5mm')
        p = g.panel(row=0, col=0, aspect=1.0)
        fig, axes = g.build(fig_width='4in')
        axes[p].set_ylabel('A long ylabel that overhangs left')
        _engine(fig).execute(fig)
        assert abs(fig.get_size_inches()[0] - 4.0) < 1e-9
        plt.close(fig)

    def test_gap_never_below_user_gap(self):
        # After execute(), the physical gap between two side-by-side panels
        # must be >= the user-specified gap (the engine may grow it, never shrink).
        user_gap_in = 3 / 25.4          # 3 mm
        g = Grid(rows=['auto'], cols=['auto', 'auto'], margin='3mm', wgap='3mm')
        pl = g.panel(row=0, col=0, aspect=1.0)
        pr = g.panel(row=0, col=1, aspect=1.0)
        fig, axes = g.build(fig_width='4in')
        axes[pl].set_ylabel('label')
        axes[pr].set_ylabel('label')
        _engine(fig).execute(fig)
        fw = fig.get_size_inches()[0]
        pos_l = axes[pl].get_position()
        pos_r = axes[pr].get_position()
        actual_gap = (pos_r.x0 - (pos_l.x0 + pos_l.width)) * fw
        assert actual_gap >= user_gap_in - 1e-6
        plt.close(fig)

    def test_large_user_gap_not_shrunk(self):
        # A user-specified gap larger than any label overhang must be preserved exactly.
        user_gap_in = 20 / 25.4         # 20 mm — deliberately large
        g = Grid(rows=['auto'], cols=['auto', 'auto'], margin='3mm', wgap='20mm')
        pl = g.panel(row=0, col=0, aspect=1.0)
        pr = g.panel(row=0, col=1, aspect=1.0)
        fig, axes = g.build(fig_width='6in')
        _engine(fig).execute(fig)       # no labels → overhang = 0
        fw = fig.get_size_inches()[0]
        pos_l = axes[pl].get_position()
        pos_r = axes[pr].get_position()
        actual_gap = (pos_r.x0 - (pos_l.x0 + pos_l.width)) * fw
        assert abs(actual_gap - user_gap_in) < 1e-4
        plt.close(fig)

    def test_cache_prevents_redundant_solves(self):
        # Calling execute() twice in a row (nothing changed) must only invoke
        # _resolve once — the second call hits the cache.
        g = Grid(rows=['auto'], cols=['auto'], margin='5mm')
        g.panel(row=0, col=0, aspect=1.0)
        fig, _ = g.build(fig_width='4in')

        call_count = [0]
        original = g._resolve
        def counting(*args, **kwargs):
            call_count[0] += 1
            return original(*args, **kwargs)
        g._resolve = counting  # type: ignore[method-assign]

        eng = _engine(fig)
        eng.execute(fig)   # first call: re-solves
        eng.execute(fig)   # second call: cache hit
        assert call_count[0] == 1
        plt.close(fig)
