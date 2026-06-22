# Changelog

## 1.4.0 - Upcoming

### Added

- Added prospective AWARE 2.0 built-in methods for country-average yearly
  characterization factors across ``SSP126``, ``SSP370``, and ``SSP585``.
  The methods include irrigation, non-irrigation, unspecified, and combined
  CPC-discriminated variants.
- Added basin-specific stochastic distributions for prospective AWARE methods,
  with basin IDs retained so values and weights can be aligned before sampling.
- Added explicit method-level interpolation metadata for prospective methods.
  Supported policies can linearly interpolate missing numeric year indices and
  use nearest-endpoint extrapolation outside the available source years.

### Changed

- Made scenario/year interpolation opt-in via method metadata. Methods without
  a supported ``interpolation`` block, such as ``SCP_1.0``, keep the legacy
  exact-or-last parameter fallback behavior.
- Updated deterministic and stochastic CF evaluation to use the declared
  interpolation policy consistently, including aggregate fallback weights and
  uncertainty sampling.

### Documentation

- Added prospective AWARE documentation, an example notebook, and regression
  coverage for deterministic interpolation, stochastic interpolation,
  nearest-endpoint extrapolation, and no-policy fallback behavior.

### Fixes

- Fixed Brightway activity lookup helpers on large databases by avoiding broad
  activity scans and improving tuple/code-based lookup paths.

### Examples

- Updated the French hydrogen AWARE example workflow to use the current
  ecoinvent 3.11 Brightway project name.

### CI

- Fixed the conda build setup in GitHub Actions so the package under test is
  installed reliably.

### Geography

- Bundled IAM and ecoinvent topology mappings derived from premise so
  non-ecoinvent regions such as ``OAS``, ``SSA``, ``LAM``, ``CAZ``,
  ``EUR``, ``REF``, ``MEA``, ``NEU``, ``CHA``, and ecoinvent IAI aggregate
  regions can participate in geographic fallback resolution.
- Registered bundled topology files under separate namespaces to avoid silent
  overwrites where different IAM models use the same region names with
  different country memberships. Bare region lookups now merge matching
  namespaced definitions.
- Suppressed noisy ``Geomatcher`` and ``country_converter`` lookup diagnostics
  during normal failed geography probes, keeping notebook output focused on
  Edges progress messages and actionable warnings.
- Added geography aliases for ``World -> GLO`` and ``US-PR -> PR``.

## 1.3.0 - 2026-05-18

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
