# edges/georesolver.py

from functools import lru_cache
import logging
from constructive_geometries import Geomatcher
from .utils import load_missing_geographies, get_str

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class GeoResolver:
    """
    Resolve geographic containment/coverage using constructive_geometries + project weights.

    :param weights: Mapping of (supplier_loc, consumer_loc) tuples to numeric weights.
    :return: GeoResolver instance.
    """

    def __init__(self, weights: dict):
        """
        Initialize the resolver and normalize internal weight keys.

        :param weights: Mapping of (supplier_loc, consumer_loc) -> weight value.
        :return: None
        """
        self.weights = {get_str(k): v for k, v in weights.items()}
        self._weights_keys_sorted = tuple(sorted(self.weights.keys()))
        self.weights_key = ",".join(sorted(self.weights.keys()))
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Dependencies from constructive_geometries and your utils
        self.geo = Geomatcher()
        self.missing_geographies = load_missing_geographies()

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

        available_set = set(weights_available)  # NEW: O(1) membership, stable
        if exceptions:
            exceptions = tuple(get_str(e) for e in exceptions)
        exc_set = set(exceptions or ())

        if location in self.missing_geographies:
            for e in self.missing_geographies[location]:
                e_str = get_str(e)
                if (
                    e_str in available_set
                    and e_str != location
                    and e_str not in exc_set
                ):
                    results.append(e_str)
        else:
            method = "contained" if containing else "within"
            raw = []
            try:
                for e in getattr(self.geo, method)(
                    location,
                    biggest_first=False,
                    exclusive=containing,
                    include_self=False,
                ):
                    e_str = get_str(e)
                    raw.append(e_str)
            except KeyError:
                self.logger.info("Region %s: no geometry found.", location)

            if containing:
                # We want *contained* regions (per your interpretation). Keep all that pass filters.
                for e_str in raw:
                    if (
                        e_str in available_set
                        and e_str != location
                        and e_str not in exc_set
                    ):
                        results.append(e_str)
            else:
                # We want the *containers* (regions that contain `location`).
                # Collect all valid containers, then choose the single most specific deterministically.
                containers = [
                    e_str
                    for e_str in raw
                    if (
                        e_str in available_set
                        and e_str != location
                        and e_str not in exc_set
                    )
                ]
                if containers:
                    # Choose container with the fewest leaves; tie-break by code.
                    best = min(containers, key=lambda r: (self._leaf_count(r), r))
                    results = [best]
                else:
                    results = []

        # Deduplicate and enforce deterministic ordering (harmless for 0/1 element)
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
            weights_available=self._weights_keys_sorted,
            containing=containing,
            exceptions=exceptions,
        )

    @lru_cache(maxsize=8192)
    def _leaf_count(self, region: str) -> int:
        """Number of leaf countries contained in `region` (large for unknown regions)."""
        try:
            return len(self.geo.contained(region, include_self=False, leaves=True))
        except KeyError:
            return 10**9  # unknown = worst (least specific)

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
