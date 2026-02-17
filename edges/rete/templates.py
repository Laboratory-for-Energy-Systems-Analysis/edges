from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

MatrixType = Literal["biosphere", "technosphere"]


@dataclass
class ReteNode:
    id: int
    name: str | None = None
    location: str | None = None
    reference_product: str | None = None
    suppliers: list[int] = field(default_factory=list)


@dataclass
class ReteRuleSide:
    matrix: MatrixType
    name: str | None = None
    location: str | None = None
    reference_product: str | None = None
    operator: str = "equals"
    excludes: list[str] | None = None


@dataclass
class ReteRule:
    id: int
    supplier: ReteRuleSide
    consumer: ReteRuleSide
    value: float | str
    uncertainty: dict | None = None
