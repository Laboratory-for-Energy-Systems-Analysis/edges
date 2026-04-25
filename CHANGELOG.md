# Changelog

## 1.3.0 - Upcoming

### Security

- Hardened symbolic CF expression evaluation by validating expressions with a
  narrow AST allowlist before evaluation. Supported expressions remain focused
  on arithmetic, parameter names, literals, and bare allowlisted function calls
  such as ``GWP(...)``.
- Replaced the previous ``__builtins__: None`` eval sandbox with sanitized
  globals, blocking object traversal, attribute access, subscripts,
  comprehensions, lambdas, imports, f-strings, and method calls in method
  expressions.
- Documented that Python callables passed through ``allowed_functions`` are
  trusted code, and included the allowed function namespace in the expression
  cache key to avoid stale cached results across different callables.

## 1.2.9 - 2026-04-19

### Mixed supplier methods

- Added support for LCIA methods that combine ``biosphere`` and
  ``technosphere`` supplier matrices in one JSON file. Core workflows now
  support mixed ``biosphere-technosphere`` and
  ``technosphere-technosphere`` characterization in the same run.

### Reporting

- Added ``EdgeLCIA.generate_cf_table(split_aggregate_consumers=True)`` for
  deterministic runs. Weighted fallback rows for aggregate or dynamic consumer
  regions are now replaced by country-specific rows using the exact shares
  stored during geographic fallback matching.
- Stored per-exchange ``reporting_split`` metadata on fallback CF entries so
  the raw country split can be inspected directly on ``cfs_mapping`` and
  deterministic ``scenario_cfs`` entries.
- Mixed-method CF tables now expose ``supplier matrix`` and ``direction``
  columns so biosphere and technosphere contributions can be distinguished in
  the exported results.

### Data

- Added dedicated IBIF v2 road methods:
  ``("IBIF", "biodiversity", "roads", "overall")`` and
  ``("IBIF", "biodiversity", "roads", "vertebrates")``.
- Extended the mixed IBIF ``all pressures`` methods for ``overall`` and
  ``vertebrates`` to include the road pressure rows alongside emissions and
  land occupation.

### Documentation

- Documented mixed supplier-method support and the new IBIF road/all-pressures
  behavior in the README, quickstart, methods guide, user guide, and the
  ``EdgeLCIA.generate_cf_table`` API docstring.

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
