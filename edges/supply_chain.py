# edge_supply_chain.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from io import StringIO
import textwrap as _tw

import math
import time
from typing import Optional, Sequence, Tuple, Dict, Any, List

import pandas as pd
import plotly.graph_objects as go


try:
    from bw2data.backends.peewee import Activity
except ImportError:  # bw2data >= 4.0
    from bw2data.backends import Activity

from .edgelcia import EdgeLCIA


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
    height_max: int = 1600,
    auto_width: bool = False,
    per_level_width: int = 250,
    per_node_width: int = 2,
    width_min: int = 900,
    width_max: Optional[int] = None,
    node_thickness: int = 18,
    node_pad: int = 12,
    lock_x_by_level: bool = True,
    balance_mode: str = "match",
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
            key = fallback_key(i, r)
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

    # Build link rows (always include below-cutoff)
    def link_rows() -> List[Tuple[int, int, float, Any, Any]]:
        out = []
        for _, r in df.iterrows():
            pid = r.get(col_parent)
            if pd.isna(pid) or pid is None:
                continue
            prow = df.loc[df[col_id] == pid]
            if prow.empty:
                continue
            parent_key = prow.iloc[0]["_node_key"]
            child_key = r["_node_key"]
            s_idx = key_to_idx.get(parent_key)
            t_idx = key_to_idx.get(child_key)
            if s_idx is None or t_idx is None:
                continue
            v = float(r[col_score] or 0.0)
            if v == 0:
                continue
            out.append((s_idx, t_idx, v, prow.iloc[0], r))
        return out

    rows_all = link_rows()

    # Incident magnitudes (for ordering/spacing)
    import collections

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

    # Figure height
    max_nodes_in_col = max((len(v) for v in nodes_by_col.values()), default=1)
    required_px = int(max_nodes_in_col * (node_thickness + node_pad) + 110)
    n_nodes = len(labels_vis)
    est_h_soft = int(
        base_height
        + per_level_px * (len(levels) - 1)
        + per_node_px * math.sqrt(max(1, n_nodes))
    )
    est_h = max(height_min, required_px, est_h_soft)
    est_h = min(est_h, height_max)

    # y positions (proportional + min-gap)
    def proportional_centers(order: List[int], lo: float, hi: float) -> List[float]:
        if not order:
            return []
        weights = [max(1e-12, magnitude[i]) for i in order]
        total = float(sum(weights))
        fracs = [w / total for w in weights]
        span = hi - lo
        centers, cum = [], 0.0
        for f in fracs:
            centers.append(lo + (cum + 0.5 * f) * span)
            cum += f
        return centers

    def enforce_min_gap(
        centers: List[float], lo: float, hi: float, min_gap: float
    ) -> List[float]:
        if not centers:
            return []
        y = centers[:]
        for i in range(1, len(y)):
            if y[i] - y[i - 1] < min_gap:
                y[i] = y[i - 1] + min_gap
        if y[-1] > hi:
            old_lo, old_hi = y[0], y[-1]
            if old_hi - old_lo < 1e-9:
                return [lo + k * (hi - lo) / (len(y) - 1) for k in range(len(y))]
            y = [lo + (p - old_lo) * (hi - lo) / (old_hi - old_lo) for p in y]
        for i in range(len(y) - 2, -1, -1):
            if y[i + 1] - y[i] < min_gap:
                y[i] = y[i + 1] - min_gap
        return [max(lo, min(hi, v)) for v in y]

    min_gap = (node_thickness + node_pad) / float(est_h)
    y_full = [0.5] * len(labels_vis)

    # Specials first in last column
    special_order_keys = [
        ("direct emissions", "__GLOBAL__"),
        ("activities below cutoff", "__GLOBAL__"),
        ("loss", "__GLOBAL__"),
    ]
    special_indices = [key_to_idx[k] for k in special_order_keys if k in key_to_idx]

    for c, idxs in nodes_by_col.items():
        if not idxs:
            continue
        lo, hi = 0.04, 0.96
        if c == last_col:
            ordered_rest = sorted(
                [i for i in idxs if i not in special_indices],
                key=lambda i: (-magnitude[i], labels_vis[i]),
            )
            ordered = special_indices + ordered_rest  # specials at top
        else:
            ordered = sorted(idxs, key=lambda i: (-magnitude[i], labels_vis[i]))
        centers0 = proportional_centers(ordered, lo, hi)
        centers = enforce_min_gap(centers0, lo, hi, min_gap)
        for i, y in zip(ordered, centers):
            y_full[i] = y

    # Parent-location palette
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

    def make_hover_link(s_idx: int, t_idx: int, v_signed: float, prow, crow) -> str:
        # % of total
        rel_total = (abs(v_signed) / abs(total_root_score)) if total_root_score else 0.0

        # locations
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

        # full names (untruncated)
        parent_name = node_full_name.get(s_idx, "")
        child_name = node_full_name.get(t_idx, "")

        # Extract reference product from our (name, ref product, location) tuple keys
        def _ref_product_from_row(row):
            ak = row.get(col_id)
            if isinstance(ak, tuple) and len(ak) >= 2:
                rp = ak[1]
                if rp is not None and not (isinstance(rp, float) and pd.isna(rp)):
                    return str(rp)
            return None

        parent_rp = _ref_product_from_row(prow)
        child_rp = _ref_product_from_row(crow)

        # Is the child a special synthetic node?
        nm_special = None
        if (
            isinstance(child_key, tuple)
            and len(child_key) == 2
            and child_key[1] == "__GLOBAL__"
        ):
            nm_special = child_key[
                0
            ]  # "direct emissions" | "activities below cutoff" | "loss"

        # Build extra RP lines
        extra_lines = []
        if parent_rp:
            extra_lines.append(f"<br><i>Parent ref product:</i> {parent_rp}")

        # Prefer a real child's RP; if special node, use our best context
        child_context = child_rp
        if not child_context and nm_special == "activities below cutoff":
            # Summary string prepared in _walk(): e.g. "H2 (40%), O2 (20%), +3 more"
            child_context = crow.get("collapsed_ref_products")

        if child_context:
            label = "Child ref product" + (
                "(s)" if nm_special == "activities below cutoff" else ""
            )
            extra_lines.append(f"<br><i>{label}:</i> {child_context}")

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
        rp = node_full_refprod.get(i, "")
        if rp:
            parts.append(f"<i>Ref. product:</i> {rp}")
        loc = node_full_loc.get(i, "")
        if loc:
            parts.append(loc)
        node_hoverdata.append("<br>".join(parts))

    # ---------- NEW: per-parent balancing factors (based on ABS widths) ----------
    # Compute in/out sums per node (using full set of rows)
    in_abs = collections.defaultdict(float)
    out_abs = collections.defaultdict(float)
    for s_idx, t_idx, v_signed, _, _ in rows_all:
        a = abs(v_signed)
        out_abs[s_idx] += a
        in_abs[t_idx] += a

    # Determine per-parent scale for outgoing links
    balance_mode = str(balance_mode).lower()
    out_scale = collections.defaultdict(lambda: 1.0)
    for node_idx in range(len(labels_vis)):
        out_sum = out_abs.get(node_idx, 0.0)
        in_sum = in_abs.get(node_idx, 0.0)
        if out_sum <= 0:
            out_scale[node_idx] = 1.0
            continue
        if balance_mode == "match" and in_sum > 0:
            out_scale[node_idx] = in_sum / out_sum  # may up- or down-scale
        elif balance_mode == "cap" and in_sum > 0:
            out_scale[node_idx] = min(1.0, in_sum / out_sum)  # only downscale
        else:
            out_scale[node_idx] = 1.0

    # Build links for both color modes, applying the per-parent scale to outgoing widths
    def links_category(rows):
        src, tgt, val, colr, hov = [], [], [], [], []
        for s_idx, t_idx, v_signed, prow, crow in rows:
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
            v = abs(v_signed) if use_abs_for_widths else abs(v_signed)
            v *= out_scale[s_idx]  # <-- balance here
            src.append(s_idx)
            tgt.append(t_idx)
            val.append(v)
            colr.append(c)
            hov.append(make_hover_link(s_idx, t_idx, v_signed, prow, crow))
        return dict(source=src, target=tgt, value=val, color=colr, customdata=hov)

    def links_by_parentloc(rows):
        src, tgt, val, colr, hov = [], [], [], [], []
        for s_idx, t_idx, v_signed, prow, crow in rows:
            base = loc_to_color.get(prow.get(col_location, "") or "—", color_other)
            c = hex_to_rgba(base, 0.60)
            v = abs(v_signed) if use_abs_for_widths else abs(v_signed)
            v *= out_scale[s_idx]  # <-- balance here
            src.append(s_idx)
            tgt.append(t_idx)
            val.append(v)
            colr.append(c)
            hov.append(make_hover_link(s_idx, t_idx, v_signed, prow, crow))
        return dict(source=src, target=tgt, value=val, color=colr, customdata=hov)

    links_cat = links_category(rows_all)
    links_loc = links_by_parentloc(rows_all)

    # Traces (two sankeys)
    def make_trace(link_dict: Dict[str, list]) -> go.Sankey:
        node_dict = dict(
            pad=node_pad,
            thickness=node_thickness,
            label=labels_vis,
            color=colors_full,
            customdata=node_hoverdata,
            hovertemplate="%{customdata}<extra></extra>",
        )
        arrangement = "fixed" if lock_x_by_level else "snap"
        if lock_x_by_level:
            node_dict["x"] = x_full
            node_dict["y"] = y_full
        return go.Sankey(
            arrangement=arrangement,
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

    # Buttons — push higher; add extra air before legend
    if add_toggle:
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.5,
                    xanchor="center",
                    y=1.30,
                    yanchor="top",
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
            ]
        )

    # Width & layout
    if auto_width:
        est_w, autosize_flag = None, True
    else:
        raw_w = per_level_width * len(levels) + per_node_width * math.sqrt(
            max(1, n_nodes)
        )
        if width_max is not None:
            raw_w = min(width_max, raw_w)
        est_w, autosize_flag = max(width_min, int(raw_w)), False

    fig.update_layout(
        height=est_h,
        width=est_w,
        autosize=autosize_flag,
        margin=dict(l=8, r=8, t=132 if add_toggle else 56, b=8),  # more top margin
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,  # a bit farther from buttons
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
        ),
    )

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
            demand={act: 1.0},
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
        self.elcia.map_aggregate_locations()
        self.elcia.map_dynamic_locations()
        self.elcia.map_contained_locations()
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

    def _walk(
        self,
        act: Activity,
        amount: float,
        level: int,
        parent: Optional[Tuple[str, str, str]],
        _precomputed_score: Optional[float] = None,  # internal: to avoid recompute
    ) -> List[SupplyChainRow]:
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
                    demand={act: amount},
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
            )
        ]

        # Depth limit
        if level >= self.level:
            self._p(
                f"{indent}[stop] reached level limit ({self.level}); returning node only"
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
                        # supplier exists but fails cutoff → goes to below
                        below_extra.append((sup, sup_amt, sup_score))

                # Any remainder (untested tail) → treat as residual under the market node
                residual = ch_score - promoted_scores
                if abs(residual) > 0:
                    # We don’t know the exact composition of the tail; attribute residual
                    # to the market activity so the below-summary can still name something.
                    below_extra.append((ch, float("nan"), residual))

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
                    collapsed_ref_products=rp_summary,  # used by hover
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
