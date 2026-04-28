# Changelog

## Unreleased

### Added
- `Grid.build()` accepts `n_iter` (default `1`). Passing `2`–`3` runs multiple
  layout passes per draw, which helps when labels shift the layout enough that
  a single measurement leaves visible gaps. The engine stops early once the
  layout has converged.

### Fixed
- Twin axes created with `twinx()` / `twiny()` are now included in the
  overhang measurement, so their tick labels no longer get clipped.
- Outer-margin expansion now considers every axis on the figure, not only those
  whose frame edge is flush with the content boundary. Content placed far
  outside its axis (e.g. `ax.text(10, 0.5, …, transform=ax.transAxes)`) now
  correctly expands the margin.

## 0.1.0 — initial release
