# edges/georesolver.py
from __future__ import annotations

from functools import lru_cache
import logging
from constructive_geometries import Geomatcher
from .utils import load_missing_geographies, get_str

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


for name in ("country_converter", "country_converter.country_converter"):
    l = logging.getLogger(name)
    # remove existing handlers
    while l.handlers:
        h = l.handlers.pop()
        try:
            h.close()
        except:
            pass
    l.propagate = False  # don’t bubble to root
    l.setLevel(logging.ERROR)  # drop WARNINGs
logging.lastResort = None


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

        if exceptions:
            exceptions = tuple(get_str(e) for e in exceptions)

        if location in self.missing_geographies:
            for e in self.missing_geographies[location]:
                e_str = get_str(e)
                if e_str in weights_available and e_str != location:
                    if not exceptions or e_str not in exceptions:
                        results.append(e_str)
        else:
            method = "contained" if containing else "within"
            raw_candidates = []
            try:
                for e in getattr(self.geo, method)(
                    location,
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
            weights_available=tuple(self.weights.keys()),
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
