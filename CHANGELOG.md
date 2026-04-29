# Changelog

## Unreleased

## 0.2.0 - update 

### Added
- `Grid.build()` accepts `n_iter` (default `1`). Passing `2`–`3` runs multiple layout passes per draw, which helps when labels shift the layout enough that a single measurement leaves visible gaps. The engine stops early once the layout has converged.

### Fixed
- Improved overhang measurement to detect more figure parts which could potentially end up outside the figure

## 0.1.0 — initial release
