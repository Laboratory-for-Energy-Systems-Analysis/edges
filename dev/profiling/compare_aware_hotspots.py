from __future__ import annotations

import argparse
import pstats
from pathlib import Path

HOTSPOTS = (
    "map_exchanges_clips",
    "map_dynamic_locations",
    "_compute_average_cf_cached",
    "compute_average_cf",
    "process_cf_list",
)


def _extract_hotspots(stats: pstats.Stats) -> dict[str, float]:
    out = {k: 0.0 for k in HOTSPOTS}
    for (filename, _lineno, funcname), (
        _cc,
        _nc,
        _tt,
        ct,
        _callers,
    ) in stats.stats.items():
        if not filename.startswith("/Users/romain/GitHub/edges/"):
            continue
        if funcname in out:
            out[funcname] += float(ct)
    return out


def _pct(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return ((new - old) / old) * 100.0


def _fmt_s(v: float) -> str:
    return f"{v:.3f}s"


def _fmt_d(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.3f}s"


def _fmt_pct(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}%"


def compare(base_prof: Path, new_prof: Path) -> None:
    base = pstats.Stats(str(base_prof))
    new = pstats.Stats(str(new_prof))

    base_hot = _extract_hotspots(base)
    new_hot = _extract_hotspots(new)

    print("== AWARE hotspot profile comparison")
    print(f"base: {base_prof}")
    print(f"new : {new_prof}")
    print()

    total_base = float(base.total_tt)
    total_new = float(new.total_tt)
    total_delta = total_new - total_base
    print(
        f"total runtime: {_fmt_s(total_base)} -> {_fmt_s(total_new)} "
        f"({_fmt_d(total_delta)}, {_fmt_pct(_pct(total_new, total_base))})"
    )
    print()

    print("hotspot | base | new | delta | delta%")
    for name in HOTSPOTS:
        b = base_hot.get(name, 0.0)
        n = new_hot.get(name, 0.0)
        d = n - b
        p = _pct(n, b)
        print(f"{name} | {_fmt_s(b)} | {_fmt_s(n)} | {_fmt_d(d)} | {_fmt_pct(p)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two AWARE cProfile runs and print compact hotspot deltas."
    )
    parser.add_argument("--base", required=True, help="Baseline .prof path")
    parser.add_argument("--new", required=True, help="New .prof path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_prof = Path(args.base)
    new_prof = Path(args.new)

    if not base_prof.exists():
        raise FileNotFoundError(f"Baseline profile not found: {base_prof}")
    if not new_prof.exists():
        raise FileNotFoundError(f"New profile not found: {new_prof}")

    compare(base_prof, new_prof)


if __name__ == "__main__":
    main()
