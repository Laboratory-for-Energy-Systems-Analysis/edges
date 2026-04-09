# Changelog

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
