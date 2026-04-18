# Changelog

## 1.2.9 - Upcoming

### Reporting

- Added ``EdgeLCIA.generate_cf_table(split_aggregate_consumers=True)`` for
  deterministic runs. Weighted fallback rows for aggregate or dynamic consumer
  regions are now replaced by country-specific rows using the exact shares
  stored during geographic fallback matching.
- Stored per-exchange ``reporting_split`` metadata on fallback CF entries so
  the raw country split can be inspected directly on ``cfs_mapping`` and
  deterministic ``scenario_cfs`` entries.

### Documentation

- Documented the new reporting option and split metadata access in the README,
  quickstart, methods guide, user guide, and the ``EdgeLCIA.generate_cf_table``
  API docstring.

## 1.1 - 2026-04-09

### GeoPolRisk - country pairs

- Updated [`edges/data/GeoPolRisk_paired_2024.json`](/Users/romain/Github/edges/edges/data/GeoPolRisk_paired_2024.json) from version `1.0` to `1.1`.
- Removed all `46` direct `GLO -> GLO` characterization-factor rows from the method.
- Kept the location-resolution logic in the configured `edges` strategy stack:
  `map_exchanges`, `map_aggregate_locations`, `map_dynamic_locations`,
  `map_contained_locations`, and `map_remaining_locations_to_global`.
- Rationale: generic `GLO -> GLO` supplier-consumer pairs should be resolved by
  the mapping strategies, especially `map_aggregate_locations()`, instead of
  being encoded directly in the base paired method file.
