from __future__ import annotations

import contextlib
from functools import lru_cache
from io import StringIO
import logging
from constructive_geometries import Geomatcher
from .utils import (
    load_builtin_topologies,
    load_legacy_geographies,
    load_missing_geographies,
    get_str,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@contextlib.contextmanager
def _silent_geomatcher_lookup():
    """Suppress stdout/stderr and country_converter log chatter during probes."""
    logger_names = ("country_converter", "country_converter.country_converter")
    loggers = [logging.getLogger(name) for name in logger_names]
    previous = [(log.disabled, log.level) for log in loggers]
    for log in loggers:
        log.disabled = True
    try:
        with (
            contextlib.redirect_stdout(StringIO()),
            contextlib.redirect_stderr(StringIO()),
        ):
            yield
    finally:
        for log, (disabled, level) in zip(loggers, previous):
            log.disabled = disabled
            log.setLevel(level)


class GeoResolver:
    """
    Resolve geographic containment/coverage using constructive_geometries + project weights.

    :param weights: Mapping of (supplier_loc, consumer_loc) tuples to numeric weights.
    :return: GeoResolver instance.
    """

    def __init__(
        self,
        weights: dict,
        additional_topologies: dict = None,
        use_builtin_topologies: bool = True,
    ):
        """
        Initialize the resolver and normalize internal weight keys.

        :param weights: Mapping of (supplier_loc, consumer_loc) -> weight value.
        :return: None
        """
        # Keep supplier/consumer keys intact when provided as tuples.
        # Backward-compatible: also accept flat string keys.
        norm_weights = {}
        for k, v in weights.items():
            if isinstance(k, tuple) and len(k) == 2:
                norm_key = (get_str(k[0]), get_str(k[1]))
            else:
                # Legacy flat location key; keep on both sides
                loc = get_str(k)
                norm_key = (loc, loc)
            norm_weights[norm_key] = v

        self.weights = norm_weights
        self.available_supplier_locations = {s for s, _ in self.weights.keys()}
        self.available_consumer_locations = {c for _, c in self.weights.keys()}
        self.available_locations = (
            self.available_supplier_locations | self.available_consumer_locations
        )
        self.weights_key = ",".join(sorted(f"{s}|{c}" for s, c in self.weights.keys()))
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Dependencies from constructive_geometries and your utils
        self.geo = Geomatcher()
        self.missing_geographies = load_missing_geographies()
        self.legacy_geographies = load_legacy_geographies()

        if use_builtin_topologies:
            for namespace, topology in load_builtin_topologies().items():
                self._add_topology_definitions(topology, namespace)

        if additional_topologies:
            self._add_topology_definitions(additional_topologies, "ecoinvent")
        self._add_topology_definitions({"World": ["GLO", "RoW"]}, "ecoinvent")

    def _normalize_location(self, location: str) -> str | None:
        """Normalize noisy legacy labels before consulting Geomatcher."""
        cleaned = " ".join(get_str(location).split()).strip().rstrip(", ")
        unresolvable = set(
            self.legacy_geographies.get("unresolvable_placeholders", []) or []
        )
        if cleaned in unresolvable:
            return None
        aliases = self.legacy_geographies.get("aliases", {}) or {}
        return aliases.get(cleaned, cleaned)

    def _clean_topology_definitions(
        self, definitions: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Apply Edges geography aliases to topology members before registration."""
        cleaned = {}
        for region, members in (definitions or {}).items():
            region_key = " ".join(get_str(region).split()).strip().rstrip(", ")
            cleaned_members = []
            for member in members or []:
                normalized = self._normalize_location(member)
                if normalized is not None:
                    cleaned_members.append(normalized)
            cleaned[region_key] = cleaned_members
        return cleaned

    def _add_topology_definitions(
        self, definitions: dict[str, list[str]], namespace: str
    ) -> None:
        """Register topology definitions without leaking lookup diagnostics."""
        cleaned = self._clean_topology_definitions(definitions)
        with _silent_geomatcher_lookup():
            try:
                self.geo.add_definitions(cleaned, namespace, relative=True)
            except KeyError as exc:
                self.logger.info(
                    "Skipping topology namespace %s because a member could not be resolved: %s",
                    namespace,
                    exc,
                )

    def _resolve_geomatcher_keys(self, location: str) -> tuple[str | tuple, ...]:
        """
        Resolve all Geomatcher keys that can represent a location string.

        constructive_geometries delegates unknown string handling to
        country_converter, which writes "not found" messages to stderr and ISO3
        fallbacks to stdout before raising or returning. Edges treats these as
        normal failed fallback candidates, so keep them out of notebook output.
        """
        keys = []

        with _silent_geomatcher_lookup():
            try:
                keys.append(self.geo._actual_key(location))
            except KeyError:
                pass

        for key in self.geo.topology:
            if isinstance(key, tuple) and len(key) >= 2 and get_str(key) == location:
                keys.append(key)

        unique = []
        seen = set()
        for key in keys:
            if key not in seen:
                unique.append(key)
                seen.add(key)
        return tuple(unique)

    def find_locations(
        self,
        location: str,
        weights_available: tuple,
        containing: bool = True,
        exceptions: tuple | None = None,
    ) -> list[str]:
        """
        Find locations that contain (or are contained by) a given location, filtered by availability.

        :param location: Base location code to resolve from.
        :param weights_available: Iterable of allowed region codes to consider.
        :param containing: If True, return regions that contain the base location; else contained regions.
        :param exceptions: Optional tuple of region codes to exclude.
        :return: List of matching region codes, filtered and ordered as discovered.
        """
        results = []

        if exceptions:
            exceptions = tuple(get_str(e) for e in exceptions)

        original_location = get_str(location)
        location = self._normalize_location(original_location)
        if location is None:
            return results

        if (
            location != original_location
            and location in weights_available
            and (not exceptions or location not in exceptions)
        ):
            results.append(location)

        if location in self.missing_geographies:
            for e in self.missing_geographies[location]:
                e_str = get_str(e)
                if e_str in weights_available and e_str != location:
                    if not exceptions or e_str not in exceptions:
                        results.append(e_str)
        else:
            resolved_locations = self._resolve_geomatcher_keys(location)
            if not resolved_locations:
                self.logger.info("Region %s: no geometry found.", location)
                return sorted(set(results))

            method = "contained" if containing else "within"
            raw_candidates = []
            try:
                for resolved_location in resolved_locations:
                    for e in getattr(self.geo, method)(
                        resolved_location,
                        biggest_first=False,
                        exclusive=containing,
                        include_self=False,
                    ):
                        e_str = get_str(e)
                        raw_candidates.append(e_str)
                        if (
                            e_str in weights_available
                            and e_str != location
                            and (not exceptions or e_str not in exceptions)
                        ):
                            results.append(e_str)
                            if not containing:
                                break
            except KeyError:
                self.logger.info("Region %s: no geometry found.", location)

        # Deduplicate and enforce deterministic ordering
        return sorted(set(results))

    @lru_cache(maxsize=2048)
    def _cached_lookup(
        self, location: str, containing: bool, exceptions: tuple | None = None
    ) -> list:
        """
        Cached backend for resolving candidate locations.

        :param location: Base location code.
        :param containing: If True, resolve containing regions; else contained regions.
        :param exceptions: Optional tuple of region codes to exclude.
        :return: List of candidate region codes.
        """
        return self.find_locations(
            location=location,
            # GeoResolver resolves candidate geographies agnostic of side.
            # Side-specific filtering is handled in resolve_candidate_locations().
            weights_available=tuple(self.available_locations),
            containing=containing,
            exceptions=exceptions,
        )

    def resolve(
        self, location: str, containing=True, exceptions: list[str] | None = None
    ) -> list:
        """
        Resolve candidate regions for a given location with caching.

        :param location: Base location code.
        :param containing: If True, resolve containing regions; else contained regions.
        :param exceptions: Optional list of region codes to exclude.
        :return: List of candidate region codes.
        """
        return self._cached_lookup(
            location=get_str(location),
            containing=containing,
            exceptions=tuple(exceptions) if exceptions else None,
        )

    def batch(
        self,
        locations: list[str],
        containing=True,
        exceptions_map: dict[str, list[str]] | None = None,
    ) -> dict[str, list[str]]:
        """
        Resolve candidate regions for multiple locations at once.

        :param locations: List of base location codes.
        :param containing: If True, resolve containing regions; else contained regions.
        :param exceptions_map: Optional mapping of location -> list of regions to exclude.
        :return: Dict mapping each input location to its list of candidate region codes.
        """
        return {
            loc: self.resolve(
                loc, containing, exceptions_map.get(loc) if exceptions_map else None
            )
            for loc in locations
        }
