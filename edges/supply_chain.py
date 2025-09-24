# edge_supply_chain.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from io import StringIO
import textwrap as _tw
import collections

import math
import time
from typing import Optional, Sequence, Tuple, Dict, Any, List

import pandas as pd
import plotly.graph_objects as go

import os
from pathlib import Path
import html
import re
import plotly.io as pio


try:
    from bw2data.backends.peewee import Activity
except ImportError:  # bw2data >= 4.0
    from bw2data.backends import Activity

from .edgelcia import EdgeLCIA

from bw2data import __version__ as bw2data_version

if isinstance(bw2data_version, str):
    bw2data_version = tuple(map(int, bw2data_version.split(".")))

if bw2data_version >= (4, 0, 0):
    is_bw25 = True
else:
    is_bw25 = False


# --- helpers for labels
def truncate_one_line(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 1)] + "…"


def make_label_two_lines(name: str, location: str, name_chars: int) -> str:
    """Line 1: truncated name (single line). Line 2: full location, never truncated."""
    n = truncate_one_line(name or "", name_chars)
    loc = "" if (location is None or pd.isna(location)) else str(location).strip()
    return f"{n}\n{loc}" if loc else n


def _is_market_name(val: Any) -> bool:
    """True if the activity 'name' looks like a market node."""
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    return s.startswith("market for ") or s.startswith("market group for ")


# --- Multi-method (multi-impact) HTML export ---------------------------------
def save_sankey_html_multi(
    label_to_df: Dict[str, pd.DataFrame],
    path: str,
    *,
    title: str = "Supply chain Sankey — multiple impact categories",
    offline: bool = True,
    auto_open: bool = True,
    plot_kwargs: Optional[Dict[str, Any]] = None,
    modebar_remove: tuple = ("lasso2d", "select2d"),
) -> str:
    """
    Save several Sankey figures (one per impact category) into a single tabbed HTML.

    Parameters
    ----------
    label_to_df : {label: DataFrame}
        Keys are tab labels (e.g., method names); values are the dataframes to plot.
    path : str
        Output file path; '.html' will be appended if missing.
    title : str
        Browser tab title.
    offline : bool
        If True, embed plotly.js once into the file. Otherwise load from CDN.
    auto_open : bool
        If True, open the file in a browser after writing.
    plot_kwargs : dict
        Extra kwargs forwarded to sankey_from_supply_df (e.g., width_max, height_max).
    modebar_remove : tuple
        Modebar buttons to remove in each figure.

    Returns
    -------
    str : the file path written.
    """

    plot_kwargs = plot_kwargs or {}
    if not path.lower().endswith(".html"):
        path += ".html"
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)

    # Build one figure per label
    pieces: List[tuple[str, str]] = []
    include = "cdn" if not offline else True
    config = {"displaylogo": False, "modeBarButtonsToRemove": list(modebar_remove)}

    def _slug(s: str) -> str:
        s2 = re.sub(r"\s+", "-", s.strip())
        s2 = re.sub(r"[^A-Za-z0-9\-_]", "", s2)
        return s2 or "tab"

    first = True
    for label, df in label_to_df.items():
        fig = sankey_from_supply_df(df, **plot_kwargs)
        # include plotly.js only once
        html_snippet = pio.to_html(
            fig,
            include_plotlyjs=(include if first else False),
            full_html=False,
            config=config,
        )
        pieces.append((label, html_snippet))
        first = False

    # Simple tab UI (CSS+JS) and body with all figures (hidden except first)
    # We wrap each snippet in a container <div class="tab-pane"> and switch display via JS.
    tabs_html = []
    panes_html = []
    for i, (label, snippet) in enumerate(pieces):
        tab_id = f"tab-{_slug(label)}"
        active = "active" if i == 0 else ""
        tabs_html.append(
            f'<button class="tab-btn {active}" onclick="showTab(\'{tab_id}\', this)">{html.escape(label)}</button>'
        )
        panes_html.append(
            f'<div id="{tab_id}" class="tab-pane {active}">{snippet}</div>'
        )

    full = f"""<!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8"/>
        <title>{html.escape(title)}</title>
        <style>
          body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; }}
          .tabs {{ position: sticky; top: 0; background: #fafafa; border-bottom: 1px solid #eee; padding: 8px; z-index: 10; display: flex; flex-wrap: wrap; gap: 6px; }}
          .tab-btn {{ border: 1px solid #ddd; background: #fff; border-radius: 6px; padding: 6px 10px; cursor: pointer; }}
          .tab-btn.active {{ background: #0d6efd; color: white; border-color: #0d6efd; }}
          .tab-pane {{ display: none; padding: 8px; }}
          .tab-pane.active {{ display: block; }}
        </style>
        </head>
        <body>
          <div class="tabs">
            {''.join(tabs_html)}
          </div>
          {''.join(panes_html)}
        <script>
          function showTab(id, btn) {{
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            const el = document.getElementById(id);
            if (el) el.classList.add('active');
            if (btn) btn.classList.add('active');
            // Force Plotly to resize when switching tabs (in case container size changed)
            if (window.Plotly && el) {{
              el.querySelectorAll('.js-plotly-plot').forEach(plot => {{
                try {{ window.Plotly.Plots.resize(plot); }} catch(e) {{}}
              }});
            }}
          }}
          // Ensure first tab active on load
          (function() {{
            const firstPane = document.querySelector('.tab-pane');
            const firstBtn = document.querySelector('.tab-btn');
            if (firstPane) firstPane.classList.add('active');
            if (firstBtn) firstBtn.classList.add('active');
          }})();
        </script>
        </body>
        </html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(full)

    if auto_open:
        import webbrowser

        webbrowser.open(f"file://{os.path.abspath(path)}")
    return path


def sankey_from_supply_df(
    df: pd.DataFrame,
    *,
    col_level: str = "level",
    col_id: str = "activity_key",
    col_parent: str = "parent_key",
    col_name: str = "name",
    col_location: str = "location",
    col_score: str = "score",
    col_amount: str = "amount",
    wrap_chars: int = 18,
    max_label_lines: int = 2,
    use_abs_for_widths: bool = True,
    add_toggle: bool = True,
    base_height: int = 380,
    per_level_px: int = 110,
    per_node_px: int = 6,
    height_min: int = 460,
    height_max: int = 1200,
    auto_width: bool = False,
    per_level_width: int = 250,
    per_node_width: int = 2,
    width_min: int = 900,
    width_max: Optional[int] = None,
    node_thickness: int = 18,
    node_pad: int = 12,
    lock_x_by_level: bool = True,
    balance_mode: str = "none",
    palette: Sequence[str] = (
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ),
    # Category colors
    color_direct: str = "#E53935",
    color_below: str = "#FB8C00",
    color_loss: str = "#FDD835",
    color_other: str = "#9E9E9E",
    col_ref_product: str = "reference product",
    enable_highlight: bool = True,
    highlight_top_k: int = 25,
    highlight_alpha_on: float = 0.9,
    highlight_alpha_off: float = 0.08,
    y_spacing: str = "linear",  # "equal" | "sqrt" | "linear" | "by_score"
    node_instance_mode: str = "merge",  # "merge" | "by_parent" | "by_child_level"
) -> go.Figure:
    """Sankey with last-level specials, untruncated hover labels, per-parent outgoing balancing, and tidy UI."""
    if df.empty:
        raise ValueError("Empty DataFrame")

    df = df.copy()

    for c in [col_level, col_name, col_score, col_id, col_parent]:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}' in df")

    if col_location not in df.columns:
        df[col_location] = ""
    else:
        df[col_location] = df[col_location].apply(
            lambda x: "" if (pd.isna(x) or x is None) else str(x)
        )

    # Root total for %
    try:
        root = df.loc[df[col_level] == df[col_level].min()].iloc[0]
        total_root_score = float(root[col_score])
    except Exception:
        total_root_score = float(df[col_score].abs().max())

    # Helpers

    def _rgba_with_alpha(c: str, a: float) -> str:
        c = str(c)
        if c.startswith("rgba("):
            # replace alpha
            parts = c[5:-1].split(",")
            if len(parts) >= 4:
                parts = [p.strip() for p in parts[:3]] + [f"{a:.3f}"]
                return f"rgba({','.join(parts)})"
        if c.startswith("#"):
            return hex_to_rgba(c, a)
        # fallback: try to parse "rgb(r,g,b)"
        if c.startswith("rgb("):
            parts = c[4:-1].split(",")
            if len(parts) == 3:
                parts = [p.strip() for p in parts]
                return f"rgba({parts[0]},{parts[1]},{parts[2]},{a:.3f})"
        # last resort: force to grey w/ alpha
        return f"rgba(150,150,150,{a:.3f})"

    def _append_tag_to_label(lbl: str, tag: str) -> str:
        """Append a short tag to the *last* line of a 1–2 line label."""
        parts = lbl.split("\n")
        if not parts:
            return tag
        parts[-1] = f"{parts[-1]}  {tag}"
        return "\n".join(parts)

    def _normalize_special(raw: Any) -> Optional[str]:
        if pd.isna(raw):
            return None
        s = str(raw).strip().lower()
        if not s:
            return None
        # Direct emissions variants
        if s.startswith("direct emissions"):
            # accept "direct emissions", "direct emissions/res. use", etc.
            return "direct emissions"
        # Below cutoff variants
        if s in {"activities below cutoff", "below cutoff"}:
            return "activities below cutoff"
        # Loss
        if s == "loss":
            return "loss"
        return None

    special_names = {
        "direct emissions": "Direct emissions/Res. use",
        "activities below cutoff": "Activities below cutoff",
        "loss": "Loss",
    }
    SPECIAL_NODE_COLOR = {
        "direct emissions": color_direct,
        "activities below cutoff": color_below,
        "loss": color_loss,
    }

    def is_special(nm: Any) -> bool:
        return _normalize_special(nm) in special_names

    def special_key(row) -> Optional[Tuple[str, str]]:
        nm = _normalize_special(row[col_name])
        return (nm, "__GLOBAL__") if nm else None

    def special_label(nm: str) -> str:
        return special_names[nm]

    def fallback_key(idx, r):
        ak = r.get(col_id)
        if pd.notna(ak):
            return ak
        return (r.get(col_name), r.get(col_location), r.get("unit"), int(idx))

    def hex_to_rgba(h: str, a: float) -> str:
        h = h.lstrip("#")
        if len(h) == 3:
            h = "".join([c * 2 for c in h])
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        return f"rgba({r},{g},{b},{a})"

    def palette_cycle(i: int, base: Sequence[str]) -> str:
        return base[i % len(base)]

    # Collect full ref-product text per node index for hover
    node_full_refprod: Dict[int, str] = {}

    def _row_refprod(row) -> str:
        """Best-effort: prefer explicit column; else try second item of activity_key tuple."""
        # explicit column
        if col_ref_product in df.columns:
            val = row.get(col_ref_product, None)
            if pd.notna(val) and val is not None and str(val).strip():
                return str(val).strip()
        # infer from activity_key tuple (name, reference product, location)
        ak = row.get(col_id, None)
        if (
            isinstance(ak, tuple)
            and len(ak) >= 2
            and pd.notna(ak[1])
            and ak[1] is not None
        ):
            s = str(ak[1]).strip()
            if s:
                return s
        return ""

    df["_is_special"] = df[col_name].apply(is_special)

    # Columns (specials live in the *last real* level)
    # Columns: compute the column set from NON-SPECIAL nodes
    # so specials don't create their own extra column.
    # (We compute df["_is_special"] above with is_special/normalizer.)
    levels_all = sorted(int(l) for l in df[col_level].unique())

    non_special_mask = ~df["_is_special"]
    levels_real = (
        sorted(int(l) for l in df.loc[non_special_mask, col_level].unique())
        if non_special_mask.any()
        else []
    )

    levels = levels_real if levels_real else levels_all

    col_index = {L: i for i, L in enumerate(levels)}
    ncols = len(levels)
    max_real_level = levels[-1]  # last level among REGULAR nodes
    last_col = col_index[max_real_level]
    level_to_color = {lvl: i for i, lvl in enumerate(levels)}

    # Build nodes (visible truncated labels + full label for hover)
    labels_vis: List[str] = []
    colors_full: List[str] = []
    x_full: List[float] = []
    key_to_idx: Dict[Any, int] = {}
    node_full_name: Dict[int, str] = {}
    node_full_loc: Dict[int, str] = {}

    node_key_by_idx: Dict[Any, Any] = {}

    def _node_key_for_row(idx, r) -> Any:
        if r["_is_special"]:
            return special_key(r)  # keep specials global

        base = fallback_key(idx, r)  # usually the activity_key tuple
        mode = (node_instance_mode or "merge").lower()

        if mode == "merge":
            return base

        elif mode == "by_parent":
            # one instance per (activity, parent)
            return ("by_parent", base, r.get(col_parent))

        elif mode == "by_child_level":
            # one instance per (activity, child level) – useful if the same activity appears at multiple levels
            return ("by_level", base, int(r[col_level]))

        else:
            # fallback to old behavior if an unknown value is passed
            return base

    for i, r in df.sort_values([col_level]).iterrows():
        L = int(r[col_level])
        full_name = str(r[col_name]) if pd.notna(r[col_name]) else ""
        full_loc = (
            str(r.get(col_location, "")) if pd.notna(r.get(col_location, "")) else ""
        )

        if r["_is_special"]:
            key = special_key(r)  # e.g., ("direct emissions","__GLOBAL__")
            label_disp = special_label(key[0])
            x_val = last_col / max(1, (ncols - 1)) if ncols > 1 else 0.0
            color = SPECIAL_NODE_COLOR.get(key[0], palette[0])
        else:
            key = _node_key_for_row(i, r)
            label_disp = make_label_two_lines(full_name, full_loc, wrap_chars)
            L_eff = L if L in col_index else max_real_level
            x_val = col_index[L_eff] / max(1, (ncols - 1)) if ncols > 1 else 0.0
            color = palette[level_to_color.get(L_eff, 0) % len(palette)]

        vis_lbl = wrap_label(label_disp, wrap_chars, max_label_lines)

        if key not in key_to_idx:
            idx = len(labels_vis)
            key_to_idx[key] = idx
            labels_vis.append(vis_lbl)
            colors_full.append(color)
            x_full.append(float(max(0.0, min(0.999, x_val))))
            node_full_name[idx] = full_name
            node_full_loc[idx] = full_loc
            node_full_refprod[idx] = "" if r["_is_special"] else _row_refprod(r)
        node_key_by_idx[i] = key

    df["_node_key"] = pd.Series(
        (node_key_by_idx.get(i) for i in df.index),
        index=df.index,
        dtype="object",
    )

    rowid_to_nodeidx = {}
    if "row_id" in df.columns:
        for _, r in df.iterrows():
            rid = r.get("row_id", None)
            if rid is not None and not pd.isna(rid):
                rowid_to_nodeidx[int(rid)] = key_to_idx[r["_node_key"]]

    # --- Per-node score/share for node hover ------------------------------------
    node_score_by_idx = collections.defaultdict(float)
    node_share_by_idx = collections.defaultdict(float)

    accumulate_for_merge = (node_instance_mode or "merge").lower() == "merge"
    for _, r in df.iterrows():
        idx = key_to_idx.get(r["_node_key"])
        if idx is None:
            continue
        sc = float(r.get(col_score) or 0.0)
        sh = (
            float(r.get("share_of_total"))
            if "share_of_total" in df.columns and pd.notna(r.get("share_of_total"))
            else ((sc / total_root_score) if total_root_score else 0.0)
        )
        if r["_is_special"] or accumulate_for_merge:
            node_score_by_idx[idx] += sc
            node_share_by_idx[idx] += sh
        else:
            node_score_by_idx[idx] = sc
            node_share_by_idx[idx] = sh

    # Build link rows (always include below-cutoff)
    def link_rows():
        out = []

        # Fast lookup by row_id if present
        df_by_rowid = None
        if "row_id" in df.columns:
            df_by_rowid = {
                int(rr["row_id"]): rr
                for _, rr in df.iterrows()
                if rr.get("row_id") is not None and not pd.isna(rr.get("row_id"))
            }

        for _, r in df.iterrows():
            s_idx, prow = None, None

            # Preferred: wire by parent_row_id (exact instance)
            pri = r.get("parent_row_id", None)
            if pri is not None and not pd.isna(pri) and df_by_rowid is not None:
                pri = int(pri)
                prow = df_by_rowid.get(pri)
                if prow is not None:
                    s_idx = rowid_to_nodeidx.get(pri)

            # Fallback: wire by parent_key (may merge instances)
            if s_idx is None:
                pid = r.get(col_parent)
                if pd.isna(pid) or pid is None:
                    continue
                prows = df.loc[df[col_id] == pid]
                if prows.empty:
                    continue
                prow = prows.iloc[0]
                parent_key = prow["_node_key"]
                s_idx = key_to_idx.get(parent_key)

            # Target (child)
            t_idx = key_to_idx.get(r["_node_key"])
            if s_idx is None or t_idx is None:
                continue

            v = float(r[col_score] or 0.0)
            if v == 0:
                continue

            out.append((s_idx, t_idx, v, prow, r))
        return out

    rows_all = link_rows()

    # --- adjacency for highlight ---------------------------------------------
    from collections import defaultdict, deque

    # rows_all: list of (s_idx, t_idx, v_signed, prow, crow)
    children = defaultdict(list)
    parents = defaultdict(list)
    for li, (s_idx, t_idx, _v, _prow, _crow) in enumerate(rows_all):
        children[s_idx].append(t_idx)
        parents[t_idx].append(s_idx)

    from collections import defaultdict as _dd

    # Indices of special global nodes
    _special_idx = {idx for key, idx in key_to_idx.items()
                    if isinstance(key, tuple) and len(key) == 2 and key[1] == "__GLOBAL__"}

    # Group by full (name, location) for non-special nodes
    _label_groups = _dd(list)
    for i in range(len(labels_vis)):
        if i in _special_idx:
            continue
        key = (node_full_name.get(i, "").strip(), node_full_loc.get(i, "").strip())
        _label_groups[key].append(i)

    _instance_info = {}
    for (_nm, _loc), idxs in _label_groups.items():
        if len(idxs) <= 1:
            continue
        for pos, idx in enumerate(sorted(idxs), start=1):
            pars = parents.get(idx, [])
            if len(pars) == 1:
                p_name = (node_full_name.get(pars[0], "") or "").strip()
                short_parent = truncate_one_line(p_name, 16) or "parent"
                tag = f"⟵ {short_parent}"
            else:
                tag = f"[{pos}]"
            labels_vis[idx] = _append_tag_to_label(labels_vis[idx], tag)
            _instance_info[idx] = (pos, len(idxs))

    def _descendants(root: int) -> set[int]:
        out, q = set([root]), deque([root])
        while q:
            u = q.popleft()
            for v in children.get(u, ()):
                if v not in out:
                    out.add(v)
                    q.append(v)
        return out

    def _ancestors(root: int) -> set[int]:
        out, q = set([root]), deque([root])
        while q:
            v = q.popleft()
            for u in parents.get(v, ()):
                if u not in out:
                    out.add(u)
                    q.append(u)
        return out

    # rank links by |value| (absolute contribution), keep top-K as candidates
    link_abs = [abs(v) for (_s, _t, v, _prow, _crow) in rows_all]
    order_links = sorted(range(len(rows_all)), key=lambda i: -link_abs[i])
    topK_idx = order_links[: min(highlight_top_k, len(order_links))]

    # Incident magnitudes (for ordering/spacing)

    magnitude = collections.defaultdict(float)
    for s, t, v, _, _ in rows_all:
        a = abs(v)
        magnitude[s] += a
        magnitude[t] += a

    # Group nodes by column
    def col_from_x(xv: float) -> int:
        return int(round(xv * max(1, (ncols - 1))))

    nodes_by_col: Dict[int, List[int]] = {c: [] for c in range(ncols)}
    for k, idx in key_to_idx.items():
        c = col_from_x(x_full[idx])
        nodes_by_col[c].append(idx)

    from collections import Counter

    parent_locs = [(prow.get(col_location, "") or "—") for _, _, _, prow, _ in rows_all]
    loc_counts = Counter(parent_locs)
    unique_locs_sorted = [k for (k, _) in loc_counts.most_common()]
    loc_to_color: Dict[str, str] = {
        loc: palette_cycle(i, palette) for i, loc in enumerate(unique_locs_sorted)
    }
    MAX_LOC_LEGEND = 8

    # Hover (full labels)
    def _fmt_pct(x) -> str:
        try:
            v = 100.0 * float(x)
        except Exception:
            return "0%"
        if v != 0 and abs(v) < 0.01:
            return "<0.01%"
        return f"{v:.2f}%"

    def _rp_from_index_or_row(node_idx: int, row) -> str:
        """Prefer the per-node cache; if empty, re-infer from the row."""
        rp = (node_full_refprod.get(node_idx) or "").strip()
        if not rp:
            # reuse the same logic you used to build node_full_refprod
            try:
                # try explicit column if present
                if col_ref_product in df.columns:
                    val = row.get(col_ref_product, None)
                    if pd.notna(val) and val is not None and str(val).strip():
                        return str(val).strip()
            except Exception:
                pass
            # fallback to activity_key tuple
            ak = row.get(col_id, None)
            if (
                isinstance(ak, tuple)
                and len(ak) >= 2
                and pd.notna(ak[1])
                and ak[1] is not None
            ):
                return str(ak[1]).strip()
        return rp

    def make_hover_link(s_idx: int, t_idx: int, v_signed: float, prow, crow) -> str:
        # % of total
        rel_total = (abs(v_signed) / abs(total_root_score)) if total_root_score else 0.0

        parent_loc = prow.get(col_location, "") or "—"
        child_key = crow["_node_key"]
        child_loc = (
            "—"
            if (
                isinstance(child_key, tuple)
                and len(child_key) == 2
                and child_key[1] == "__GLOBAL__"
            )
            else (crow.get(col_location, "") or "—")
        )

        parent_name = node_full_name.get(s_idx, "")
        child_name = node_full_name.get(t_idx, "")

        # --- Reference products (read from the per-node cache, not the row) ---
        parent_rp = _rp_from_index_or_row(s_idx, prow)
        child_rp = _rp_from_index_or_row(t_idx, crow)

        # If the child is the special "below cutoff" node, use the summary string
        nm_special = None
        if (
            isinstance(child_key, tuple)
            and len(child_key) == 2
            and child_key[1] == "__GLOBAL__"
        ):
            nm_special = child_key[
                0
            ]  # "direct emissions" | "activities below cutoff" | "loss"
            if not child_rp and nm_special == "activities below cutoff":
                child_rp = (crow.get("collapsed_ref_products") or "").strip()

        extra_lines = []
        if parent_rp:
            extra_lines.append(f"<br><i>Parent ref product:</i> {parent_rp}")
        if child_rp:
            label = "Child ref product" + (
                "(s)" if nm_special == "activities below cutoff" else ""
            )
            extra_lines.append(f"<br><i>{label}:</i> {child_rp}")

        amt = crow.get(col_amount, None)
        amt_line = (
            f"<br>Raw amount: {amt:,.5g}"
            if (amt is not None and not pd.isna(amt))
            else ""
        )

        return (
            f"<b>{child_name}</b> ← <b>{parent_name}</b>"
            f"<br><i>Child location:</i> {child_loc}"
            f"<br><i>Parent location:</i> {parent_loc}"
            f"<br>Flow: {v_signed:,.5g}"
            f"<br>Contribution of total: {_fmt_pct(rel_total)}"
            f"{amt_line}" + "".join(extra_lines)
        )

    node_hoverdata = []
    for i in range(len(labels_vis)):
        parts = [f"<b>{node_full_name.get(i,'')}</b>"]

        rp = (node_full_refprod.get(i, "") or "").strip()
        if rp:
            parts.append(f"<i>Ref. product:</i> {rp}")

        loc = (node_full_loc.get(i, "") or "").strip()
        if loc:
            parts.append(loc)

        # Add node score and share
        sc = node_score_by_idx.get(i, None)
        if sc is not None:
            parts.append(f"<i>Node score:</i> {sc:,.6g}")
            if total_root_score:
                parts.append(f"<i>Share of total:</i> {_fmt_pct(sc/total_root_score)}")

            # If this node label was disambiguated, show the instance number
            inst = _instance_info.get(i)
            if inst:
                parts.append(f"<i>Instance:</i> #{inst[0]} of {inst[1]}")

        node_hoverdata.append("<br>".join(parts))

    # ---------- Forward pass scaling: make outgoing == actually-received incoming ----------
    balance_mode = str(balance_mode).lower()

    # base (unscaled) absolute link widths
    base_vals = [abs(v) for (_s, _t, v, _pr, _cr) in rows_all]

    # index outgoing links per source, incoming links per target
    from collections import defaultdict

    out_links = defaultdict(list)
    in_links = defaultdict(list)
    for li, (s_idx, t_idx, _v, _pr, _cr) in enumerate(rows_all):
        out_links[s_idx].append(li)
        in_links[t_idx].append(li)

    # unscaled outgoing sum per node
    out_abs = {
        node: sum(base_vals[li] for li in out_links.get(node, ()))
        for node in range(len(labels_vis))
    }

    # incoming widths after upstream scaling (initialize zeros)
    incoming_scaled = [0.0] * len(labels_vis)
    out_scale = [1.0] * len(labels_vis)  # default

    # process nodes by column from left to right so parents go first
    cols_sorted = sorted(nodes_by_col.keys())
    for col in cols_sorted:
        for node in nodes_by_col[col]:
            out_sum = out_abs.get(node, 0.0)
            in_sum = incoming_scaled[node]

            if out_sum > 0:
                if balance_mode == "match" and in_sum > 0:
                    out_scale[node] = in_sum / out_sum
                elif balance_mode == "cap" and in_sum > 0:
                    out_scale[node] = min(1.0, in_sum / out_sum)
                else:
                    out_scale[node] = 1.0
            else:
                out_scale[node] = 1.0

            # propagate scaled outgoing to children
            for li in out_links.get(node, ()):
                s_idx, t_idx, _v, _pr, _cr = rows_all[li]
                incoming_scaled[t_idx] += base_vals[li] * out_scale[node]

    # Build links for both color modes, applying the per-parent scale to outgoing widths
    def links_category(rows):
        src, tgt, val, colr, hov = [], [], [], [], []
        for li, (s_idx, t_idx, v_signed, prow, crow) in enumerate(rows):
            ck = crow["_node_key"]
            nm = ck[0] if (isinstance(ck, tuple)) else ""
            if nm == "direct emissions":
                c = hex_to_rgba(color_direct, 0.55)
            elif nm == "activities below cutoff":
                c = hex_to_rgba(color_below, 0.55)
            elif nm == "loss":
                c = hex_to_rgba(color_loss, 0.55)
            else:
                c = hex_to_rgba(color_other, 0.40)

            v = base_vals[li] * out_scale[s_idx]  # <--- scaled width
            src.append(s_idx)
            tgt.append(t_idx)
            val.append(v)
            colr.append(c)
            hov.append(make_hover_link(s_idx, t_idx, v_signed, prow, crow))
        return dict(source=src, target=tgt, value=val, color=colr, customdata=hov)

    def links_by_parentloc(rows):
        src, tgt, val, colr, hov = [], [], [], [], []
        for li, (s_idx, t_idx, v_signed, prow, crow) in enumerate(rows):
            base = loc_to_color.get(prow.get(col_location, "") or "—", color_other)
            c = hex_to_rgba(base, 0.60)
            v = base_vals[li] * out_scale[s_idx]  # <--- scaled width
            src.append(s_idx)
            tgt.append(t_idx)
            val.append(v)
            colr.append(c)
            hov.append(make_hover_link(s_idx, t_idx, v_signed, prow, crow))
        return dict(source=src, target=tgt, value=val, color=colr, customdata=hov)

    links_cat = links_category(rows_all)
    links_loc = links_by_parentloc(rows_all)

    # ----- Build "hide specials" variants (transparent color ONLY; keep values) -----
    is_special_target = []
    for _s, _t, _v, _prow, crow in rows_all:
        ck = crow["_node_key"]
        is_special_target.append(
            isinstance(ck, tuple) and len(ck) == 2 and ck[1] == "__GLOBAL__"
        )

    def _hide_colors(colors, mask):
        return [_rgba_with_alpha(c, 0.0) if m else c for c, m in zip(colors, mask)]

    cat_cols_hide = _hide_colors(links_cat["color"], is_special_target)
    loc_cols_hide = _hide_colors(links_loc["color"], is_special_target)

    # (optional) also mute hover on hidden links:
    cat_hover_hide = [
        ("" if m else h) for h, m in zip(links_cat["customdata"], is_special_target)
    ]
    loc_hover_hide = [
        ("" if m else h) for h, m in zip(links_loc["customdata"], is_special_target)
    ]

    # (optional) if you also want to fade the special nodes themselves:
    # node_colors_hide = node_colors_base[:]
    # for i in special_indices:
    #     node_colors_hide[i] = _rgba_with_alpha(node_colors_hide[i], 0.0)

    # --- base color arrays for restyling --------------------------------------
    node_colors_base = [_rgba_with_alpha(c, 1.0) for c in colors_full]
    link_colors_cat_base = list(links_cat["color"])
    link_colors_loc_base = list(links_loc["color"])

    def _make_highlight_state(link_i: int):
        """Return (node_colors, link_colors_cat, link_colors_loc) for a selected link."""
        s_idx, t_idx, _v, _prow, _crow = rows_all[link_i]

        # upstream: all ancestors of source; downstream: all descendants of target
        up_nodes = _ancestors(s_idx)
        down_nodes = _descendants(t_idx)
        on_nodes = up_nodes | down_nodes | {s_idx, t_idx}

        # choose links that stay within the upstream DAG or the downstream subtree
        on_links_mask = [False] * len(rows_all)
        for j, (sj, tj, _vj, _pr, _cr) in enumerate(rows_all):
            if (
                (sj in up_nodes and tj in up_nodes)
                or (sj in down_nodes and tj in down_nodes)
                or (j == link_i)
            ):
                on_links_mask[j] = True

        # nodes: keep hue, change alpha
        node_cols = [
            _rgba_with_alpha(
                colors_full[i],
                highlight_alpha_on if i in on_nodes else highlight_alpha_off,
            )
            for i in range(len(colors_full))
        ]
        # links: keep hue, change alpha
        link_cols_cat = [
            _rgba_with_alpha(
                link_colors_cat_base[j],
                highlight_alpha_on if on_links_mask[j] else highlight_alpha_off,
            )
            for j in range(len(rows_all))
        ]
        link_cols_loc = [
            _rgba_with_alpha(
                link_colors_loc_base[j],
                highlight_alpha_on if on_links_mask[j] else highlight_alpha_off,
            )
            for j in range(len(rows_all))
        ]
        return node_cols, link_cols_cat, link_cols_loc

    # ---- Top/bottom domain for nodes (keep nodes away from menus) -------------
    needs_top_bar = add_toggle or (enable_highlight and len(rows_all) > 0)
    top_margin = 156 if needs_top_bar else (132 if add_toggle else 56)
    bottom_margin = 8
    top_dom, bot_dom = 0.04, 0.96  # y-domain used by the sankey traces
    dom_span = bot_dom - top_dom

    # ---- Layout numbers (top/bottom margins + domain) ----
    needs_top_bar = add_toggle or (enable_highlight and len(rows_all) > 0)
    top_margin = 156 if needs_top_bar else (132 if add_toggle else 56)
    bottom_margin = 8
    top_dom, bot_dom = 0.04, 0.96
    dom_span = bot_dom - top_dom

    # Soft height heuristic
    n_nodes_total = sum(len(v) for v in nodes_by_col.values())
    est_h_soft = int(
        base_height
        + per_level_px * (len(levels) - 1)
        + per_node_px * math.sqrt(max(1, n_nodes_total))
    )
    est_h = min(height_max, max(height_min, est_h_soft))
    pane_h = max(1.0, est_h - (top_margin + bottom_margin))
    px_per_dom = pane_h * dom_span
    pad_dom = node_pad / px_per_dom
    # --- Node rectangle thickness (pixels) and its domain equivalent ---
    th_eff = int(node_thickness)  # pixels: what Plotly will actually draw
    th_norm = th_eff / max(
        1e-9, px_per_dom
    )  # domain units: min height each node occupies

    # --- Use the same scaled link values Plotly will render ---
    scaled_vals = [
        base_vals[li] * out_scale[s_idx]
        for li, (s_idx, t_idx, _v, _pr, _cr) in enumerate(rows_all)
    ]
    incoming = collections.defaultdict(float)
    outgoing = collections.defaultdict(float)
    for li, (s_idx, t_idx, _v, _pr, _cr) in enumerate(rows_all):
        v = scaled_vals[li]
        outgoing[s_idx] += v
        incoming[t_idx] += v

    # raw value-height per node (same units as link values)
    h_raw = [0.0] * len(labels_vis)
    for i in range(len(labels_vis)):
        h_raw[i] = max(incoming.get(i, 0.0), outgoing.get(i, 0.0), 1e-12)

    # per-column sums (raw)
    col_sum_raw = {c: sum(h_raw[i] for i in idxs) for c, idxs in nodes_by_col.items()}

    # global values→domain scale (limiting column sets the scale)
    if nodes_by_col:
        S_dom_candidates = []
        for c, idxs in nodes_by_col.items():
            total = col_sum_raw.get(c, 0.0)
            if total > 0:
                n = len(idxs)
                S_dom_candidates.append((dom_span - max(0, n - 1) * pad_dom) / total)
        S_dom = max(0.0, min(S_dom_candidates) if S_dom_candidates else 1.0)
    else:
        S_dom = 1.0

    # node heights in domain units
    h_dom = [h * S_dom for h in h_raw]
    # Use at least the visual rectangle height when packing to avoid overlap
    h_draw_dom = [max(h, th_norm) for h in h_dom]

    def pack_column_tops(order, lo, hi):
        if not order:
            return []
        avail = max(1e-9, hi - lo)
        n = len(order)
        total_h = sum(h_draw_dom[i] for i in order)
        total_with_pad = total_h + max(0, n - 1) * pad_dom
        pad_eff = (
            pad_dom
            if total_with_pad <= avail
            else max(0.0, (avail - total_h) / max(1, n - 1))
        )
        slack = avail - (total_h + max(0, n - 1) * pad_eff)

        # give bigger nodes a bit more breathing room
        weights = [max(1e-12, h_dom[i]) for i in order]
        wsum = sum(weights)
        gaps = [slack * (w / wsum) for w in weights]

        ytops, cur = [], lo
        for k, i in enumerate(order):
            cur += gaps[k]
            ytops.append(cur)
            cur += h_draw_dom[i] + pad_eff
        # clamp
        return [max(lo, min(hi - h_draw_dom[i], y)) for y, i in zip(ytops, order)]

    def _tie_break_key(i: int) -> tuple:
        return (-magnitude[i], labels_vis[i], i)

    # build y_top using actual heights; pin specials first in last col
    y_top = [0.5] * len(labels_vis)
    special_order_keys = [
        ("direct emissions", "__GLOBAL__"),
        ("activities below cutoff", "__GLOBAL__"),
        ("loss", "__GLOBAL__"),
    ]
    special_indices = [key_to_idx[k] for k in special_order_keys if k in key_to_idx]

    for c, idxs in nodes_by_col.items():
        if not idxs:
            continue
        lo, hi = top_dom, bot_dom
        if c == last_col:
            ordered_rest = sorted(
                [i for i in idxs if i not in special_indices], key=_tie_break_key
            )
            ordered = special_indices + ordered_rest
        else:
            ordered = sorted(idxs, key=_tie_break_key)
        y_col = pack_column_tops(ordered, lo, hi)
        for i, y in zip(ordered, y_col):
            y_top[i] = y

    # numerical guard
    EPS = 1e-6
    for c, idxs in nodes_by_col.items():
        col = sorted(idxs, key=lambda i: y_top[i])
        for k in range(1, len(col)):
            prev, cur = col[k - 1], col[k]
            min_top = y_top[prev] + h_draw_dom[prev] - EPS
            if y_top[cur] < min_top:
                y_top[cur] = min_top
        overflow = (y_top[col[-1]] + h_draw_dom[col[-1]] - bot_dom) if col else 0.0
        if overflow > 0:
            for i in col:
                y_top[i] = max(top_dom, y_top[i] - overflow)

    # Traces (two sankeys)
    th_eff = int(node_thickness)

    def make_trace(link_dict: Dict[str, list]) -> go.Sankey:
        node_dict = dict(
            pad=node_pad,
            thickness=th_eff,
            label=labels_vis,
            color=colors_full,
            customdata=node_hoverdata,
            hovertemplate="%{customdata}<extra></extra>",
        )
        arrangement = "fixed" if lock_x_by_level else "snap"
        if lock_x_by_level:
            node_dict["x"] = x_full
            node_dict["y"] = y_top  # TOP coords in domain units
        return go.Sankey(
            arrangement=arrangement,
            domain=dict(
                x=[0, 1], y=[top_dom, bot_dom]
            ),  # <--- keep nodes inside this band
            node=node_dict,
            link=dict(
                source=link_dict["source"],
                target=link_dict["target"],
                value=link_dict["value"],
                color=link_dict["color"],
                customdata=link_dict["customdata"],
                hovertemplate="%{customdata}<extra></extra>",
            ),
        )

    fig = go.Figure(data=[make_trace(links_cat), make_trace(links_loc)])
    fig.data[0].visible = True
    fig.data[1].visible = False

    # Legends
    legend_cat = [
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=color_direct),
            name="Direct emissions/Res. use",
            showlegend=True,
            hoverinfo="skip",
        ),
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=color_below),
            name="Activities below cutoff",
            showlegend=True,
            hoverinfo="skip",
        ),
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=color_loss),
            name="Loss",
            showlegend=True,
            hoverinfo="skip",
        ),
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=color_other),
            name="Other flows",
            showlegend=True,
            hoverinfo="skip",
        ),
    ]
    top_locs = unique_locs_sorted[:MAX_LOC_LEGEND]
    legend_loc = [
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=loc_to_color[loc]),
            name=f"{loc}",
            showlegend=True,
            hoverinfo="skip",
        )
        for loc in top_locs
    ]
    if len(unique_locs_sorted) > MAX_LOC_LEGEND:
        legend_loc.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color_other),
                name="Other locations",
                showlegend=True,
                hoverinfo="skip",
            )
        )
    for tr in legend_cat + legend_loc:
        fig.add_trace(tr)

    cat_legend_count = len(legend_cat)
    loc_legend_count = len(legend_loc)

    def vis_array(mode: str) -> List[bool]:
        base = [mode == "cat", mode == "loc"]
        cat_leg = [mode == "cat"] * cat_legend_count
        loc_leg = [mode == "loc"] * loc_legend_count
        return base + cat_leg + loc_leg

    # Apply initial vis
    for i, v in enumerate(vis_array("cat")):
        fig.data[i].visible = v

    # ---------------- Place the top controls without overlap ----------------
    # ---------- Build highlight dropdown buttons ----------
    highlight_buttons = []
    if enable_highlight and len(rows_all) > 0:
        # Reset option
        highlight_buttons.append(
            dict(
                label="Highlight: None",
                method="restyle",
                args=[
                    {
                        "node.color": [node_colors_base, node_colors_base],
                        "link.color": [link_colors_cat_base, link_colors_loc_base],
                    },
                    [0, 1],
                ],
            )
        )
        # Top-K links
        for rank, li in enumerate(topK_idx, start=1):
            s_idx, t_idx, _v_signed, _prow, _crow = rows_all[li]
            parent_name = node_full_name.get(s_idx, "")
            child_name = node_full_name.get(t_idx, "")
            label_txt = f"#{rank}  {child_name} ← {parent_name}"

            node_cols, link_cols_cat, link_cols_loc = _make_highlight_state(li)
            highlight_buttons.append(
                dict(
                    label=label_txt[:80],
                    method="restyle",
                    args=[
                        {
                            "node.color": [node_cols, node_cols],
                            "link.color": [link_cols_cat, link_cols_loc],
                        },
                        [0, 1],
                    ],
                )
            )

    menus = []

    # Left: color-mode buttons
    if add_toggle:
        menus.append(
            dict(
                type="buttons",
                direction="left",
                x=0.01,
                xanchor="left",  # left edge
                y=1.28,
                yanchor="top",  # above plot area
                pad=dict(l=6, r=6, t=2, b=2),
                buttons=[
                    dict(
                        label="Color: Category",
                        method="update",
                        args=[{"visible": vis_array("cat")}],
                    ),
                    dict(
                        label="Color: Parent location",
                        method="update",
                        args=[{"visible": vis_array("loc")}],
                    ),
                ],
            )
        )

    # Right: highlight dropdown (only if enabled and we have links)
    if enable_highlight and len(rows_all) > 0:
        menus.append(
            dict(
                type="dropdown",
                direction="down",
                x=0.99,
                xanchor="right",  # right edge
                y=1.28,
                yanchor="top",
                showactive=True,
                pad=dict(l=6, r=6, t=2, b=2),
                buttons=highlight_buttons,
            )
        )

    # Right/center-left: flows toggle (show/hide links to special nodes)
    menus.append(
        dict(
            type="buttons",
            direction="left",
            x=0.32,
            xanchor="left",
            y=1.28,
            yanchor="top",
            pad=dict(l=6, r=6, t=2, b=2),
            buttons=[
                dict(
                    label="Flows: Show specials",
                    method="restyle",
                    args=[
                        {
                            "link.color": [links_cat["color"], links_loc["color"]],
                            # Optional: also restore hover text
                            "link.customdata": [
                                links_cat["customdata"],
                                links_loc["customdata"],
                            ],
                        },
                        [0, 1],
                    ],
                ),
                dict(
                    label="Flows: Hide specials",
                    method="restyle",
                    args=[
                        {
                            "link.color": [cat_cols_hide, loc_cols_hide],
                            # Optional: blank hover on hidden links
                            "link.customdata": [cat_hover_hide, loc_hover_hide],
                        },
                        [0, 1],
                    ],
                ),
            ],
        )
    )

    fig.update_layout(updatemenus=menus)

    # ---------------- Layout/margins (extra top space for the controls) -----------
    needs_top_bar = add_toggle or (enable_highlight and len(rows_all) > 0)
    top_margin = 156 if needs_top_bar else (132 if add_toggle else 56)

    # ---------- Width & autosize ----------
    if auto_width:
        est_w, autosize_flag = None, True
    else:
        raw_w = per_level_width * len(levels) + per_node_width * math.sqrt(
            max(1, n_nodes_total)
        )
        if width_max is not None:
            raw_w = min(width_max, raw_w)
        est_w, autosize_flag = max(width_min, int(raw_w)), False

    fig.update_layout(
        height=est_h,
        width=est_w,
        autosize=autosize_flag,
        margin=dict(l=8, r=8, t=top_margin, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.10,  # slightly lower so it won't collide with menus
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    compact = (est_w is not None) and (est_w < 1100)
    if compact and menus:
        for m in menus:
            m.update(x=0.5, xanchor="center")
        y_base, y_step = 1.34, 0.08
        for i, m in enumerate(menus):
            m.update(y=y_base - i * y_step)
        top_margin = max(top_margin, 200)
        fig.update_layout(margin=dict(l=8, r=8, t=top_margin, b=8))

    return fig


def wrap_label(text: str, max_chars: int, max_lines: int) -> str:
    """Wrap text to at most `max_lines` lines of width `max_chars`,
    adding an ellipsis on the last line if truncated. Never breaks words/hyphens.
    """
    if not text:
        return ""

    s = str(text).strip()
    if not s:
        return ""
    lines = _tw.wrap(s, width=max_chars, break_long_words=False, break_on_hyphens=False)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if len(lines[-1]) >= max_chars:
            lines[-1] = lines[-1][: max_chars - 1] + "…"
        else:
            lines[-1] += "…"
    return "\n".join(lines)


def save_sankey_html(
    fig: go.Figure,
    path: str,
    *,
    title: str = "Supply chain Sankey",
    offline: bool = True,
    auto_open: bool = True,
    modebar_remove: tuple = ("lasso2d", "select2d"),
) -> str:
    """
    Save a Plotly Sankey figure as a standalone HTML file.

    Parameters
    ----------
    fig : go.Figure
        Figure returned by sankey_from_supply_df(...) or SupplyChain.plot_sankey(...).
    path : str
        Output file path. '.html' will be added if missing.
    title : str
        <title> of the HTML document (browser tab name).
    offline : bool
        If True, embed plotly.js inside the HTML (bigger file, fully offline).
        If False, load plotly.js from CDN (smaller file).
    auto_open : bool
        If True, open the file in a browser after writing.
    modebar_remove : tuple
        Modebar buttons to remove.

    Returns
    -------
    str
        The (possibly extended) file path that was written.
    """

    if not path.lower().endswith(".html"):
        path += ".html"

    out_dir = os.path.dirname(path) or "."
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    include = True if offline else "cdn"
    config = {
        "displaylogo": False,
        "modeBarButtonsToRemove": list(modebar_remove),
        # you can add "toImageButtonOptions":{"scale":2} if you want bigger PNG exports
    }

    # Keep figure layout as-is; just write it out
    try:
        pio.write_html(
            fig,
            file=path,
            include_plotlyjs=include,
            full_html=True,
            auto_open=auto_open,
            title=title,  # available in Plotly >=5.0
            config=config,
        )
    except TypeError:
        # Fallback for older Plotly that doesn't support 'title' in write_html
        pio.write_html(
            fig,
            file=path,
            include_plotlyjs=include,
            full_html=True,
            auto_open=auto_open,
            config=config,
        )
    return path


def save_html_multi_methods_for_activity(
    activity: Activity,
    methods: Sequence[tuple],
    path: str,
    *,
    amount: float = 1.0,
    level: int = 3,
    cutoff: float = 0.01,
    cutoff_basis: str = "total",
    scenario: str | None = None,
    scenario_idx: int | str = 0,
    use_distributions: bool = False,
    iterations: int = 100,
    random_seed: int | None = None,
    redo_flags: Optional[Dict[str, bool]] = None,
    collapse_markets: bool = False,
    plot_kwargs: Optional[Dict[str, Any]] = None,
    offline: bool = False,
    auto_open: bool = False,
    label_fn=lambda m: " / ".join(str(x) for x in m),
) -> str:
    """
    Compute one Sankey per impact method and save them into a single tabbed HTML.

    Usage:
        save_html_multi_methods_for_activity(
            activity, methods, "outputs/multi_impact.html",
            level=3, cutoff=0.01, collapse_markets=True,
            plot_kwargs=dict(width_max=1800, height_max=800),
            offline=False, auto_open=True
        )
    """
    label_to_df: Dict[str, pd.DataFrame] = {}
    for m in methods:
        sc = SupplyChain(
            activity=activity,
            method=m,
            amount=amount,
            level=level,
            cutoff=cutoff,
            cutoff_basis=cutoff_basis,
            scenario=scenario,
            scenario_idx=scenario_idx,
            use_distributions=use_distributions,
            iterations=iterations,
            random_seed=random_seed,
            redo_flags=redo_flags,
            collapse_markets=collapse_markets,
        )
        sc.bootstrap()
        df, _, _ = sc.calculate()
        label_to_df[label_fn(m)] = df

    return save_sankey_html_multi(
        label_to_df,
        path,
        plot_kwargs=plot_kwargs or {},
        offline=offline,
        auto_open=auto_open,
        title="Multi-impact Sankey",

    )


@dataclass
class SupplyChainRow:
    level: int
    share_of_total: float
    score: float
    amount: float
    name: str | None
    location: str | None
    unit: str | None
    activity_key: Tuple[str, str, str] | None
    parent_key: Tuple[str, str, str] | None
    collapsed_ref_products: str | None = None
    row_id: int | None = None          # <---
    parent_row_id: int | None = None   # <---


class SupplyChain:

    def __init__(
        self,
        activity: Activity,
        method: tuple,
        *,
        amount: float = 1.0,
        level: int = 3,
        cutoff: float = 0.01,
        cutoff_basis: str = "total",
        scenario: str | None = None,
        scenario_idx: int | str = 0,
        use_distributions: bool = False,
        iterations: int = 100,
        random_seed: int | None = None,
        redo_flags: Optional[Dict[str, bool]] = None,
        collapse_markets: bool = False,
        debug: bool = False,
        dbg_max_prints: int = 2000,
        market_top_k: int = 60,
    ):
        if not isinstance(activity, Activity):
            raise TypeError("`activity` must be a Brightway2 Activity.")

        self.root = activity
        self.method = method
        self.amount = float(amount) * (
            -1.0 if self._is_waste_process(activity) else 1.0
        )
        self.level = int(level)
        self.cutoff = float(cutoff)
        self.cutoff_basis = str(cutoff_basis).lower()
        if self.cutoff_basis not in {"total", "parent"}:
            raise ValueError("cutoff_basis must be 'total' or 'parent'")

        self.scenario = scenario
        self.scenario_idx = scenario_idx
        self.collapse_markets = bool(collapse_markets)

        self.elcia = EdgeLCIA(
            demand={activity: self.amount},
            method=method,
            use_distributions=use_distributions,
            iterations=iterations,
            random_seed=random_seed,
            scenario=scenario,
        )

        # Control which fallback mapping passes are allowed during recursion
        self._redo_flags = dict(
            run_aggregate=True, run_dynamic=True, run_contained=True, run_global=True
        )
        if redo_flags:
            self._redo_flags.update(redo_flags)

        self._total_score: Optional[float] = None
        self._unit_score_cache: Dict[Any, float] = {}
        self._market_flat_cache: Dict[Any, List[Tuple[Activity, float]]] = {}

        # --- Debugging ---
        self.debug = bool(debug)
        self._dbg_prints = 0
        self._dbg_max_prints = int(dbg_max_prints)
        self._dbg_muted = False

        self.market_top_k = int(market_top_k)

        self._row_counter = 0

    def _next_row_id(self) -> int:
        rid = self._row_counter
        self._row_counter += 1
        return rid

    # ---------- Public API ---------------------------------------------------

    # ---------- Debug helpers ----------
    def _p(self, msg: str):
        """Rate-limited debug print."""
        if not self.debug:
            return
        if self._dbg_prints < self._dbg_max_prints:
            print(msg)
            self._dbg_prints += 1
        elif not self._dbg_muted:
            print("[debug muted after limit]")
            self._dbg_muted = True

    @staticmethod
    def _short_act(act: Activity) -> str:
        try:
            nm = str(act.get("name"))
        except Exception:
            nm = "<?>"
        try:
            loc = act.get("location")
        except Exception:
            loc = None
        locs = f" [{loc}]" if loc else ""
        return nm + locs

    @staticmethod
    def _is_market_name(val: Any) -> bool:
        """Return True if an activity name looks like an ecoinvent market."""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return False
        s = str(val).strip().lower()
        return s.startswith("market for ") or s.startswith("market group for ")

    def _flatten_market_suppliers(
        self, market_act: Activity
    ) -> List[Tuple[Activity, float]]:
        """Flatten a MARKET into final suppliers with per-unit coefficients. Cached."""
        mk = self._act_cache_key(market_act)
        hit = self._market_flat_cache.get(mk)
        if hit is not None:
            self._p(
                f"[flatten] cache HIT for market: {self._short_act(market_act)} → {len(hit)} suppliers"
            )
            return hit

        t0 = time.perf_counter()
        out_pairs: List[Tuple[Activity, float]] = []
        nodes_visited = 0
        edges_traversed = 0

        def _dfs(act: Activity, coef: float, path: set):
            nonlocal nodes_visited, edges_traversed
            nodes_visited += 1
            ak = self._act_cache_key(act)
            if ak in path:
                self._p(
                    f"[flatten] cycle @ {self._short_act(act)} (coef {coef:.4g}) → treat as terminal"
                )
                out_pairs.append((act, coef))
                return
            if not _is_market_name(act.get("name")):
                out_pairs.append((act, coef))
                return
            sups = list(act.technosphere())
            if not sups:
                out_pairs.append((act, coef))
                return
            path.add(ak)
            for ex in sups:
                sup = ex.input
                amt = float(ex["amount"])
                edges_traversed += 1
                _dfs(sup, coef * amt, path)
            path.remove(ak)

        self._p(
            f"[flatten] MISS for market: {self._short_act(market_act)} — expanding…"
        )
        _dfs(market_act, 1.0, set())

        # aggregate duplicates
        from collections import defaultdict

        agg: Dict[Any, float] = defaultdict(float)
        key2act: Dict[Any, Activity] = {}
        for s, c in out_pairs:
            k = self._act_cache_key(s)
            agg[k] += c
            key2act[k] = s

        flat = [(key2act[k], agg[k]) for k in agg]
        self._market_flat_cache[mk] = flat
        dt = time.perf_counter() - t0
        self._p(
            f"[flatten] built {len(flat)} suppliers (visited nodes={nodes_visited}, edges={edges_traversed}) "
            f"in {dt:.3f}s for market {self._short_act(market_act)}"
        )
        # If this is insanely large, say so:
        if len(flat) > 2000:
            self._p(
                f"[flatten][WARN] very large supplier set: {len(flat)} for {self._short_act(market_act)}"
            )
        return flat

    def _score_per_unit(self, act: Activity) -> float:
        """Memoized unit score for an activity (uses current scenario/flags)."""
        k = self._act_cache_key(act)
        hit = self._unit_score_cache.get(k)
        if hit is not None:
            return hit
        t0 = time.perf_counter()

        self.elcia.redo_lcia(
            demand={act.id if is_bw25 else act: 1.0},
            scenario_idx=self.scenario_idx,
            scenario=self.scenario,
            recompute_score=True,
            **self._redo_flags,
        )
        s = float(self.elcia.score or 0.0)
        dt = time.perf_counter() - t0
        self._unit_score_cache[k] = s
        self._p(
            f"[unit-score] MISS {self._short_act(act)} → {s:.5g} in {dt:.3f}s (cache size={len(self._unit_score_cache)})"
        )
        if dt > 0.5:
            self._p(f"[unit-score][SLOW] {self._short_act(act)} took {dt:.3f}s")
        return s

    @staticmethod
    def _act_cache_key(act: Activity) -> Any:
        # Prefer unique, stable identifiers if available
        for attr in ("id", "key"):
            if hasattr(act, attr):
                return getattr(act, attr)
        # Fallback: use db/code if present, else your tuple key
        try:
            return (act["database"], act["code"])
        except Exception:
            return (act["name"], act.get("reference product"), act.get("location"))

    def bootstrap(self) -> float:
        """
        Run the initial EdgeLCIA pipeline on the root demand to build CM,
        then compute and store the total score.
        """
        # Standard pipeline on root demand
        self.elcia.lci()
        self.elcia.map_exchanges()
        if self._redo_flags.get("run_aggregate", True):
            self.elcia.map_aggregate_locations()

        if self._redo_flags.get("run_dynamic", True):
            self.elcia.map_dynamic_locations()

        if self._redo_flags.get("run_contained", True):
            self.elcia.map_contained_locations()

        if self._redo_flags.get("run_global", True):
            self.elcia.map_remaining_locations_to_global()

        self.elcia.evaluate_cfs(scenario_idx=self.scenario_idx, scenario=self.scenario)
        self.elcia.lcia()
        self._total_score = float(self.elcia.score or 0.0)
        return self._total_score

    def calculate(self) -> tuple[pd.DataFrame, float, float]:
        """
        Recursively traverse the technosphere, returning (df, total_score, reference_amount).
        Call `bootstrap()` first for best performance/coverage.
        """
        if self._total_score is None:
            self.bootstrap()
        rows = self._walk(self.root, self.amount, level=0, parent=None)
        df = pd.DataFrame([asdict(r) for r in rows])
        return df, float(self._total_score or 0.0), self.amount

    def as_text(self, df: pd.DataFrame) -> StringIO:
        """Pretty text view of the breakdown."""
        buf = StringIO()
        if df.empty:
            buf.write("No contributions (total score is 0?)\n")
            return buf
        view = df[
            ["level", "share_of_total", "score", "amount", "name", "location", "unit"]
        ].copy()
        view["share_of_total"] = (view["share_of_total"] * 100).round(2)
        view["score"] = view["score"].astype(float).round(6)
        view["amount"] = view["amount"].astype(float)
        with pd.option_context("display.max_colwidth", 60):
            buf.write(view.to_string(index=False))
        return buf

    # ---------- Internals ----------------------------------------------------

    def _walk(self, act, amount, level, parent, _precomputed_score=None, _parent_row_id=None):

        """Traverse one node with lazy market expansion (expand only above-cutoff, top-K)."""
        indent = "  " * level
        self._p(
            f"{indent}→ ENTER level {level}: {self._short_act(act)}  amount={amount:.5g}"
        )

        # --- Node score ---
        t0 = time.perf_counter()
        if level == 0:
            node_score = float(self._total_score or 0.0)
            self._p(f"{indent}[score] root uses precomputed total={node_score:.5g}")
        else:
            if _precomputed_score is None:
                self.elcia.redo_lcia(
                    demand={ (act.id if is_bw25 else act): amount },
                    scenario_idx=self.scenario_idx,
                    scenario=self.scenario,
                    recompute_score=True,
                    **self._redo_flags,
                )
                node_score = float(self.elcia.score or 0.0)
                dt = time.perf_counter() - t0
                self._p(
                    f"{indent}[score] redo_lcia {self._short_act(act)} → {node_score:.5g} in {dt:.3f}s"
                )
            else:
                node_score = float(_precomputed_score)
                self._p(f"{indent}[score] reused child precomputed={node_score:.5g}")

        total = float(self._total_score or 0.0)
        share = (node_score / total) if total != 0 else 0.0
        cur_key = self._key(act)

        # Cycle guard
        if parent is not None and cur_key == parent:
            self._p(f"{indent}[cycle] parent == current; emitting LOSS and returning")
            return [
                SupplyChainRow(
                    level=level,
                    share_of_total=share,
                    score=node_score,
                    amount=float(amount),
                    name="loss",
                    location=None,
                    unit=None,
                    activity_key=None,
                    parent_key=parent,
                )
            ]

        rid = self._next_row_id()
        rows: List[SupplyChainRow] = [
            SupplyChainRow(
                level=level,
                share_of_total=share,
                score=node_score,
                amount=float(amount),
                name=act["name"],
                location=act.get("location"),
                unit=act.get("unit"),
                activity_key=cur_key,
                parent_key=parent,
                row_id=rid,  # <---
                parent_row_id=_parent_row_id,  # <---
            )
        ]

        # Depth limit
        if level >= self.level:
            self._p(f"{indent}[stop] reached level limit ({self.level}); returning node only")
            return rows

        # Treat unknown-amount nodes as terminals
        if isinstance(amount, float) and math.isnan(amount):
            if node_score != 0.0:
                rows.append(
                    SupplyChainRow(
                        level=level + 1,
                        share_of_total=(node_score / total) if total else 0.0,
                        score=node_score,
                        amount=float("nan"),
                        name="Direct emissions/Res. use",
                        location=None,
                        unit=None,
                        activity_key=None,
                        parent_key=cur_key,
                        row_id=self._next_row_id(),
                        parent_row_id=rid,
                    )
                )
            return rows

        # ----------------------------------------------------------------------
        # 1) Collect children WITHOUT expanding markets; aggregate & score once.
        # ----------------------------------------------------------------------
        from collections import defaultdict

        agg_amounts: Dict[Any, float] = defaultdict(float)
        key_to_act: Dict[Any, Activity] = {}

        def _add_child(a: Activity, amt: float):
            k = self._act_cache_key(a)
            agg_amounts[k] += amt
            key_to_act[k] = a

        exs = list(act.technosphere())
        for exc in exs:
            ch = exc.input
            ch_amt = amount * float(exc["amount"])
            _add_child(ch, ch_amt)

        # Score each unique child ONCE with unit scores
        children: List[Tuple[Activity, float, float]] = []
        t_score0 = time.perf_counter()
        for k, amt in agg_amounts.items():
            a = key_to_act[k]
            unit = self._score_per_unit(a)
            children.append((a, amt, unit * amt))
        dt_score = time.perf_counter() - t_score0
        self._p(
            f"{indent}[collect+score] unique_children={len(children)} in {dt_score:.3f}s (unit-cache={len(self._unit_score_cache)})"
        )

        if not children:
            # Leaf → all is direct emissions
            if node_score != 0.0:
                rows.append(
                    SupplyChainRow(
                        level=level + 1,
                        share_of_total=(node_score / total) if total else 0.0,
                        score=node_score,
                        amount=float("nan"),
                        name="Direct emissions/Res. use",
                        location=None,
                        unit=None,
                        activity_key=None,
                        parent_key=cur_key,
                        row_id=self._next_row_id(),
                        parent_row_id=rid,
                    )
                )
            self._p(f"{indent}[leaf] no children; returning")
            return rows

        # --- Cutoff split (track BOTH above and below) -----------------------
        denom_parent = abs(node_score)
        denom_total = abs(total)
        denom_for_cutoff = (
            denom_parent
            if (self.cutoff_basis == "parent" and denom_parent > 0)
            else denom_total
        )
        self._p(
            f"{indent}[cutoff] basis={'parent' if denom_for_cutoff==denom_parent else 'total'} "
            f"cutoff={self.cutoff:.4g} denom={denom_for_cutoff:.5g}"
        )

        # Keep explicit lists; we’ll need `below` later to summarize ref products
        above: List[Tuple[Activity, float, float]] = []
        below: List[Tuple[Activity, float, float]] = []

        for ch, ch_amt, ch_score in children:
            rel = (abs(ch_score) / denom_for_cutoff) if denom_for_cutoff > 0 else 0.0
            if rel >= self.cutoff:
                above.append((ch, ch_amt, ch_score))
            else:
                below.append((ch, ch_amt, ch_score))

        self._p(f"{indent}[cutoff] pre-expand above={len(above)}  below={len(below)}")

        # --- Lazy market expansion (only for above-cutoff markets) ----------
        if self.collapse_markets and above:
            above_final: List[Tuple[Activity, float, float]] = []
            below_extra: List[Tuple[Activity, float, float]] = []
            K = max(0, self.market_top_k)

            for ch, ch_amt, ch_score in above:
                # Non-market stays as-is
                if not self._is_market_name(ch.get("name")):
                    above_final.append((ch, ch_amt, ch_score))
                    continue

                # Expand market into suppliers
                t_flat = time.perf_counter()
                flat = self._flatten_market_suppliers(ch)  # [(sup_act, coef_per_unit)]
                dt_flat = time.perf_counter() - t_flat
                self._p(
                    f"{indent}[expand-market] {self._short_act(ch)} suppliers={len(flat)} "
                    f"in {dt_flat:.3f}s, top_k={K}, ch_score={ch_score:.5g}"
                )

                # Rank candidates; compute scores only for top-K
                flat_sorted = sorted(flat, key=lambda t: abs(t[1]), reverse=True)

                promoted_scores = 0.0
                promoted_cnt = 0
                tested_cnt = 0

                for sup, coef in flat_sorted[:K]:
                    sup_amt = ch_amt * coef
                    unit = self._score_per_unit(sup)
                    sup_score = unit * sup_amt
                    tested_cnt += 1
                    rel = (
                        (abs(sup_score) / denom_for_cutoff)
                        if denom_for_cutoff > 0
                        else 0.0
                    )
                    if rel >= self.cutoff:
                        above_final.append((sup, sup_amt, sup_score))
                        promoted_scores += sup_score
                        promoted_cnt += 1
                    else:
                        below_extra.append((sup, sup_amt, sup_score))

                residual = ch_score - promoted_scores

                if promoted_cnt == 0:
                    # No supplier cleared the global cutoff → keep the market itself visible
                    # (don’t demote the whole thing into "below cutoff")
                    above_final.append((ch, ch_amt, ch_score))
                else:
                    # We promoted some suppliers. Decide what to do with the residual:
                    # if the residual itself is big enough, keep it visible; else send to 'below'.
                    rel_resid = (
                        (abs(residual) / denom_for_cutoff)
                        if denom_for_cutoff > 0
                        else 0.0
                    )
                    if rel_resid >= self.cutoff:
                        # Use 0.0 (not NaN) so recursion yields direct = node_score and shows a direct-emissions link
                        above_final.append((ch, 0.0, residual))
                    elif abs(residual) > 0:
                        below_extra.append((ch, 0.0, residual))  # harmless either way (we don't recurse into "below")

                self._p(
                    f"{indent}[expand-market] promoted={promoted_cnt}/{tested_cnt}  "
                    f"promoted_score={promoted_scores:.5g}  residual→below={residual:.5g}"
                )

            # Replace above with expanded set; extend below with what fell short
            above = above_final
            below.extend(below_extra)

        # --- Balance & specials ---------------------------------------------
        sum_above = sum(cs for _, _, cs in above)
        sum_below = sum(cs for _, _, cs in below)
        direct = node_score - (sum_above + sum_below)

        self._p(
            f"{indent}[balance] node={node_score:.5g} sum_above={sum_above:.5g} "
            f"sum_below={sum_below:.5g} direct={direct:.5g}"
        )

        if abs(direct) > 0.0:
            rows.append(
                SupplyChainRow(
                    level=level + 1,
                    share_of_total=(direct / total) if total else 0.0,
                    score=direct,
                    amount=float("nan"),
                    name="Direct emissions/Res. use",
                    location=None,
                    unit=None,
                    activity_key=None,
                    parent_key=cur_key,
                    row_id=self._next_row_id(),
                    parent_row_id=rid,
                )
            )

        if abs(sum_below) > 0.0:
            # Build a compact summary of ref products among below-cutoff children
            from collections import defaultdict

            agg_rp = defaultdict(float)
            for ch, _amt, cs in below:
                rp = ch.get("reference product") or ""
                agg_rp[rp] += abs(cs)

            TOPN = 6
            total_abs = sum(agg_rp.values()) or 0.0
            items = sorted(agg_rp.items(), key=lambda kv: kv[1], reverse=True)
            if total_abs > 0:
                parts = [
                    f"{(k or '—')} ({v/total_abs*100:.1f}%)" for k, v in items[:TOPN]
                ]
            else:
                parts = [(k or "—") for k, _ in items[:TOPN]]
            more = max(0, len(items) - TOPN)
            if more:
                parts.append(f"+{more} more")
            rp_summary = ", ".join(parts)

            rows.append(
                SupplyChainRow(
                    level=level + 1,
                    share_of_total=(sum_below / total) if total else 0.0,
                    score=sum_below,
                    amount=float("nan"),
                    name="activities below cutoff",
                    location=None,
                    unit=None,
                    activity_key=None,
                    parent_key=cur_key,
                    collapsed_ref_products=rp_summary,
                    row_id=self._next_row_id(),
                    parent_row_id=rid,
                )
            )

        # --- Recurse into the final above-cutoff set ------------------------
        max_list = 6
        for idx, (ch, ch_amt, ch_score) in enumerate(above):
            if idx < max_list:
                self._p(
                    f"{indent}[recurse] → child {idx+1}/{len(above)} {self._short_act(ch)} "
                    f"amt={ch_amt:.5g} score={ch_score:.5g}"
                )
            elif idx == max_list:
                self._p(
                    f"{indent}[recurse] … {len(above)-max_list} more children (muted)"
                )
            rows.extend(
                self._walk(
                    ch,
                    ch_amt,
                    level=level + 1,
                    parent=cur_key,
                    _precomputed_score=ch_score,
                    _parent_row_id=rid,
                )
            )

        self._p(
            f"{indent}← EXIT level {level}: {self._short_act(act)}  produced_rows={len(rows)}"
        )
        return rows

    # ---------- Small helpers ------------------------------------------------

    @staticmethod
    def _is_waste_process(activity: Activity) -> bool:
        for exc in activity.production():
            if exc["amount"] < 0:
                return True
        return False

    @staticmethod
    def _key(a: Activity) -> Tuple[str, str, str]:
        return (a["name"], a.get("reference product"), a.get("location"))

    def plot_sankey(self, df: pd.DataFrame, **kwargs):
        """Convenience method: EdgeSupplyChainScorer.plot_sankey(df, ...)."""
        return sankey_from_supply_df(df, **kwargs)

    def save_html(self, df: pd.DataFrame, path: str, **plot_kwargs) -> str:
        """
        Build the Sankey from `df` with plot kwargs, then save to HTML.
        Returns the final file path.
        """
        fig = self.plot_sankey(df, **plot_kwargs)
        return save_sankey_html(fig, path)
