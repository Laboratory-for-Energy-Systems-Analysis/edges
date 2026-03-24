from types import SimpleNamespace

from edges.matching_signatures import find_duplicate_clips_signatures
from edges.rete.adapter import map_exchanges_clips
from edges.rete.engine import ClipsEngine


def test_find_duplicate_clips_signatures_respects_unit():
    groups = find_duplicate_clips_signatures(
        [
            {
                "supplier": {
                    "matrix": "biosphere",
                    "name": "Water",
                    "categories": ("water",),
                    "unit": "m3",
                },
                "consumer": {"matrix": "technosphere", "location": "GLO"},
                "value": -42.95,
            },
            {
                "supplier": {
                    "matrix": "biosphere",
                    "name": "Water",
                    "categories": ("water",),
                    "unit": "m3",
                },
                "consumer": {"matrix": "technosphere", "location": "GLO"},
                "value": -0.04295,
            },
            {
                "supplier": {
                    "matrix": "biosphere",
                    "name": "Water",
                    "categories": ("water",),
                    "unit": "kg",
                },
                "consumer": {"matrix": "technosphere", "location": "GLO"},
                "value": -0.04295,
            },
        ]
    )

    assert len(groups) == 1
    assert groups[0]["indices"] == (0, 1)
    assert groups[0]["values"] == (-42.95, -0.04295)


def test_clips_template_includes_unit_slot():
    built = []
    engine = ClipsEngine.__new__(ClipsEngine)
    engine.env = SimpleNamespace(build=built.append)

    engine._build_template("bio_node")

    assert "(slot unit (type LEXEME))" in built[0]


def test_clips_rule_includes_unit_constraints():
    engine = ClipsEngine.__new__(ClipsEngine)

    rule_txt = engine._build_rule(
        {
            "id": 0,
            "supplier": {
                "matrix": "biosphere",
                "name": "Water",
                "unit": "m3",
            },
            "consumer": {
                "matrix": "technosphere",
                "location": "GLO",
                "unit": "kg",
            },
            "value": 1.0,
        }
    )[0]

    assert '(unit "m3")' in rule_txt
    assert '(unit "kg")' in rule_txt


def test_map_exchanges_clips_accepts_unit_field(monkeypatch):
    captured = {}

    class _FakeEngine:
        def run(self, data, *, rules_signature, on_match):
            captured["data"] = data
            on_match(0, 1, 2)
            return []

    monkeypatch.setattr("edges.rete.adapter._get_or_create_engine", lambda _: _FakeEngine())

    lcia = SimpleNamespace(
        method="demo",
        raw_cfs_data=[
            {
                "supplier": {
                    "matrix": "biosphere",
                    "name": "Water",
                    "categories": ("water",),
                    "unit": "m3",
                },
                "consumer": {
                    "matrix": "technosphere",
                    "location": "GLO",
                    "unit": "kg",
                },
                "value": -42.95,
            }
        ],
        technosphere_flows=[{"position": 2, "location": "GLO", "unit": "kg"}],
        biosphere_flows=[
            {
                "position": 1,
                "name": "Water",
                "categories": ("water",),
                "unit": "m3",
            }
        ],
        technosphere_edges=set(),
        biosphere_edges={(1, 2)},
        cfs_mapping=[],
        _seen_positions=set(),
        applied_strategies=[],
        _ensure_filtered_lookups_for_current_edges=lambda: None,
        _initialize_weights=lambda: None,
        _update_unprocessed_edges=lambda: None,
    )

    map_exchanges_clips(lcia)

    assert len(lcia.cfs_mapping) == 1
    assert lcia.cfs_mapping[0]["supplier"]["unit"] == "m3"
    assert lcia.cfs_mapping[0]["consumer"]["unit"] == "kg"
    assert captured["data"].bio_nodes[0]["unit"] == "m3"
    assert captured["data"].tech_nodes[0]["unit"] == "kg"
