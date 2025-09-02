# edges/georesolver.py

from functools import lru_cache
import logging
from constructive_geometries import Geomatcher
from .utils import load_missing_geographies, get_str


class GeoResolver:
    def __init__(self, weights: dict):
        self.weights = {get_str(k): v for k, v in weights.items()}
        self.weights_key = ",".join(sorted(self.weights.keys()))
        self.logger = logging.getLogger("GeoResolver")
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            fh = logging.FileHandler("georesolver.log")
            formatter = logging.Formatter(
                "%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"
            )
            fh.setFormatter(formatter)
            fh.setLevel(logging.DEBUG)
            self.logger.addHandler(fh)
            self.logger.propagate = False

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
        Find the locations containing or contained by a given location.
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
                self.logger.info(f"Region: {location}. No geometry found.")

        return results

    @lru_cache(maxsize=2048)
    def _cached_lookup(
        self, location: str, containing: bool, exceptions: tuple | None = None
    ) -> list:
        return self.find_locations(
            location=location,
            weights_available=tuple(self.weights.keys()),
            containing=containing,
            exceptions=exceptions,
        )

    def resolve(
        self, location: str, containing=True, exceptions: list[str] | None = None
    ) -> list:
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
        return {
            loc: self.resolve(
                loc, containing, exceptions_map.get(loc) if exceptions_map else None
            )
            for loc in locations
        }
