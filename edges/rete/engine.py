from __future__ import annotations

from io import StringIO
import logging
from dataclasses import dataclass
import tempfile
import os
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ReteExecutionInput:
    rules: list[dict[str, Any]]
    bio_nodes: list[dict[str, Any]]
    tech_nodes: list[dict[str, Any]]
    edges: list[tuple[int, int]]


class ClipsEngine:
    """
    Thin CLIPSpy engine facade used by the adapter.

    The full RETE rule compilation/runtime is intentionally isolated here so
    integration can evolve without touching ``EdgeLCIA`` internals.
    """

    def __init__(self):
        try:
            import clips  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            raise RuntimeError(
                "CLIPSpy backend requested but clipspy is not available."
            ) from exc
        self._clips = clips
        self.env = clips.Environment()

    def run(
        self,
        data: ReteExecutionInput,
        on_match: Callable[[int, int, int], None],
    ) -> list[tuple[str, int, int]]:
        """
        Execute RETE matching and call ``on_match(rule_id, supplier_id, consumer_id)``.

        Returns unique location rejects as ``(kind, supplier_id, consumer_id)``,
        where ``kind`` is ``"bio"`` or ``"tech"``.
        """
        rules_by_id = {int(rule["id"]): rule for rule in data.rules}
        nodes_by_id = {
            int(node["id"]): node for node in (data.bio_nodes + data.tech_nodes)
        }

        def add_result(rule_id, supplier_id, consumer_id):
            rid = int(rule_id)
            sid = int(supplier_id)
            cid = int(consumer_id)
            if rid not in rules_by_id:
                return
            if sid not in nodes_by_id or cid not in nodes_by_id:
                return
            on_match(rid, sid, cid)

        self.env.define_function(add_result)

        self._build_template("tech_node")
        self._build_template("bio_node")
        self._build_template("loc_reject")

        self._add_rules(data.rules)
        self._bulk_assert_facts(data.tech_nodes, "tech_node")
        self._bulk_assert_facts(data.bio_nodes, "bio_node")

        self.env.run()
        return self._collect_location_rejects()

    def _build_template(self, template_name: str):
        if template_name == "loc_reject":
            template = """(deftemplate loc_reject
  (slot kind (type SYMBOL))
  (slot sid (type INTEGER))
  (slot cid (type INTEGER))
)"""
            self.env.build(template)
            return

        template = f"""(deftemplate {template_name}
  (slot id (type INTEGER))
  (slot name (type LEXEME))
  (slot location (type LEXEME))
  (slot reference_product (type LEXEME))
  (slot categories_path (type LEXEME))
  (multislot class_prefixes (type STRING))
  (multislot bio_suppliers (type INTEGER))
  (multislot tech_suppliers (type INTEGER))
)"""
        self.env.build(template)

    def _collect_location_rejects(self) -> list[tuple[str, int, int]]:
        out: list[tuple[str, int, int]] = []
        for fact in self.env.facts():
            if fact.template.name != "loc_reject":
                continue
            out.append((str(fact["kind"]), int(fact["sid"]), int(fact["cid"])))
        return out

    def _bulk_assert_facts(self, nodes: list[dict[str, Any]], template_name: str):
        if not nodes:
            return
        buff = StringIO()
        for n in nodes:
            buff.write(self._node_fact(n, template_name))
            buff.write("\n")
        self._bulk_load(buff.getvalue(), facts=True)

    @staticmethod
    def _safe_lexeme(val: Any) -> str:
        if val is None:
            return "nil"
        s = str(val).strip()
        if not s:
            return "nil"
        s = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{s}"'

    def _node_fact(self, node: dict[str, Any], template_name: str) -> str:
        bio_suppliers = node.get("bio_suppliers") or []
        tech_suppliers = node.get("tech_suppliers") or []
        bio_suppliers_txt = " ".join(str(int(s)) for s in bio_suppliers)
        tech_suppliers_txt = " ".join(str(int(s)) for s in tech_suppliers)
        class_prefixes = node.get("_class_prefixes") or []
        class_prefixes_txt = " ".join(self._safe_lexeme(x) for x in class_prefixes)
        return (
            f"({template_name} "
            f"(id {int(node['id'])}) "
            f"(name {self._safe_lexeme(node.get('name'))}) "
            f"(location {self._safe_lexeme(node.get('location'))}) "
            f"(reference_product {self._safe_lexeme(node.get('reference product'))}) "
            f"(categories_path {self._safe_lexeme(node.get('_categories_path'))}) "
            f"(class_prefixes {class_prefixes_txt}) "
            f"(bio_suppliers {bio_suppliers_txt}) "
            f"(tech_suppliers {tech_suppliers_txt})"
            f")"
        )

    def _add_rules(self, rules: list[dict[str, Any]]):
        buff = StringIO()
        for r in rules:
            for rule_txt in self._build_rule(r):
                buff.write(rule_txt)
                buff.write("\n")
        self._bulk_load(buff.getvalue(), facts=False)

    @staticmethod
    def _to_safe_var_suffix(s: str) -> str:
        return re.sub(r"\W+", "_", s).strip("_") or "x"

    @staticmethod
    def _to_clips_test(var_name: str, operator: str, pattern: str, neg: bool = False):
        pattern = pattern.replace("\\", "\\\\").replace('"', '\\"')
        if operator == "contains":
            op = "eq" if neg else "neq"
            return f'({op} (str-index "{pattern}" {var_name}) FALSE)'
        if operator == "startswith":
            op = "neq" if neg else "eq"
            return f'({op} (str-index "{pattern}" {var_name}) 1)'
        raise ValueError(f"Unsupported operator '{operator}' in CLIPS backend.")

    @staticmethod
    def _rule_value(rule_side: dict[str, Any], field_name: str):
        if field_name == "reference_product":
            return rule_side.get("reference product")
        if field_name == "categories_path":
            return rule_side.get("categories")
        return rule_side.get(field_name)

    @staticmethod
    def _normalize_categories(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return "|||".join(str(v) for v in value)
        return str(value)

    @staticmethod
    def _iter_classification_pairs(value: Any) -> list[tuple[str, str]]:
        if not value:
            return []
        pairs: list[tuple[str, str]] = []
        if isinstance(value, dict):
            for scheme, codes in value.items():
                seq = codes if isinstance(codes, (list, tuple, set)) else [codes]
                for c in seq:
                    cc = str(c).split(":", 1)[0].strip()
                    if cc:
                        pairs.append((str(scheme).lower().strip(), cc))
            return pairs

        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    scheme, codes = item
                    if isinstance(codes, (list, tuple, set)):
                        for c in codes:
                            cc = str(c).split(":", 1)[0].strip()
                            if cc:
                                pairs.append((str(scheme).lower().strip(), cc))
                    else:
                        cc = str(codes).split(":", 1)[0].strip()
                        if cc:
                            pairs.append((str(scheme).lower().strip(), cc))
            return pairs

        return pairs

    def _field_slot(self, field_name: str, rule_side: dict[str, Any]) -> str | None:
        operator = rule_side.get("operator", "equals")
        target = self._rule_value(rule_side, field_name)
        excludes = rule_side.get("excludes") or []
        tests = []

        if field_name == "categories_path":
            target = self._normalize_categories(target)

        if target not in (None, "__ANY__"):
            if operator == "equals" and not excludes:
                return f"({field_name} {self._safe_lexeme(target)})"
            var = f"?{field_name}_{self._to_safe_var_suffix(str(target))}"
            if operator == "equals":
                tests.append(f"(eq {var} {self._safe_lexeme(target)})")
            else:
                tests.append(self._to_clips_test(var, operator, str(target), neg=False))
        else:
            var = f"?{field_name}_v"

        for exc in excludes:
            tests.append(self._to_clips_test(var, "contains", str(exc), neg=True))

        if not tests:
            return None
        if len(tests) == 1:
            cond = tests[0]
        else:
            cond = f"(and {' '.join(tests)})"
        return f"({field_name} {var} &:{cond})"

    def _classification_slot(self, rule_side: dict[str, Any]) -> str | None:
        pairs = self._iter_classification_pairs(rule_side.get("classifications"))
        if not pairs:
            return None

        tokens = sorted({f"{scheme}|{code}" for scheme, code in pairs if code})
        if not tokens:
            return None

        tests = " ".join(
            f"(neq (member$ {self._safe_lexeme(token)} $?cp) FALSE)" for token in tokens
        )
        if len(tokens) == 1:
            condition = tests
        else:
            condition = f"(or {tests})"
        return f"(class_prefixes $?cp &:{condition})"

    def _node_pattern(
        self,
        rule_side: dict[str, Any],
        is_supplier: bool,
        supplier_matrix: str | None = None,
        include_location: bool = True,
        location_bind_var: str | None = None,
    ) -> str:
        matrix = str(rule_side.get("matrix", "technosphere")).strip().lower()
        template = "bio_node" if matrix == "biosphere" else "tech_node"
        node_id = "?from_id" if is_supplier else "?to_id"

        parts = [f"({template} (id {node_id})"]
        if not is_supplier:
            source_matrix = (supplier_matrix or "technosphere").strip().lower()
            slot = "bio_suppliers" if source_matrix == "biosphere" else "tech_suppliers"
            parts.append(f"({slot} $? ?from_id $?)")

        for field_name in ("location", "name", "reference_product", "categories_path"):
            if field_name == "location" and not include_location:
                continue
            slot = self._field_slot(field_name, rule_side)
            if slot:
                parts.append(slot)
        if (not include_location) and location_bind_var:
            parts.append(f"(location {location_bind_var})")
        cls_slot = self._classification_slot(rule_side)
        if cls_slot:
            parts.append(cls_slot)

        parts.append(")")
        return " ".join(parts)

    def _location_match_expr(
        self, var_name: str, rule_side: dict[str, Any]
    ) -> str | None:
        target = self._rule_value(rule_side, "location")
        excludes = rule_side.get("excludes") or []
        operator = rule_side.get("operator", "equals")
        tests: list[str] = []

        if target not in (None, "__ANY__"):
            if operator == "equals":
                tests.append(f"(eq {var_name} {self._safe_lexeme(target)})")
            else:
                tests.append(
                    self._to_clips_test(var_name, operator, str(target), neg=False)
                )

        for exc in excludes:
            tests.append(self._to_clips_test(var_name, "contains", str(exc), neg=True))

        if not tests:
            return None
        if len(tests) == 1:
            return tests[0]
        return f"(and {' '.join(tests)})"

    def _build_rule(self, rule: dict[str, Any]) -> list[str]:
        rule_id = int(rule["id"])
        supplier_side = dict(rule.get("supplier") or {})
        consumer_side = dict(rule.get("consumer") or {})
        supplier_matrix = (
            str(supplier_side.get("matrix", "technosphere")).strip().lower()
        )

        supplier_pattern = self._node_pattern(
            supplier_side,
            is_supplier=True,
            include_location=False,
            location_bind_var="?s_loc",
        )
        consumer_pattern = self._node_pattern(
            consumer_side,
            is_supplier=False,
            supplier_matrix=supplier_matrix,
            include_location=False,
            location_bind_var="?c_loc",
        )
        loc_tests = []
        s_loc_match = self._location_match_expr("?s_loc", supplier_side)
        if s_loc_match:
            loc_tests.append(s_loc_match)
        c_loc_match = self._location_match_expr("?c_loc", consumer_side)
        if c_loc_match:
            loc_tests.append(c_loc_match)
        if not loc_tests:
            loc_match_expr = "TRUE"
        elif len(loc_tests) == 1:
            loc_match_expr = loc_tests[0]
        else:
            loc_match_expr = f"(and {' '.join(loc_tests)})"

        if loc_tests:
            direction_symbol = "bio" if supplier_matrix == "biosphere" else "tech"
            rhs = (
                f"(if {loc_match_expr}\n"
                f"   then\n"
                f"    (add_result {rule_id} ?from_id ?to_id)\n"
                f"   else\n"
                f"    (assert (loc_reject (kind {direction_symbol}) (sid ?from_id) (cid ?to_id))))"
            )
        else:
            rhs = f"(add_result {rule_id} ?from_id ?to_id)"

        rule_txt = (
            f"(defrule R{rule_id}\n"
            f"  {supplier_pattern}\n"
            f"  {consumer_pattern}\n"
            f"  =>\n"
            f"  {rhs}\n"
            f")"
        )
        return [rule_txt]

    def _bulk_load(self, text: str, facts: bool):
        tmp_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+", delete=False, suffix=".facts" if facts else ".clp"
            ) as f:
                f.write(text)
                tmp_file = f.name
            if facts:
                self.env.load_facts(tmp_file)
            else:
                self.env.load(tmp_file)
        finally:
            if tmp_file and os.path.exists(tmp_file):
                os.remove(tmp_file)
