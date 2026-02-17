
Theory
======

Overview
--------

The `edges` framework introduces a novel modeling layer in LCIA: **exchange-resolved LCIA**. Unlike traditional LCIA that assigns impacts to flows regardless of context, `edges` characterizes **exchanges** — the relationships between activities — allowing for CFs to reflect supplier/consumer location, activity classification, and scenario conditions.

This method is practical, scalable, and avoids the need for full GIS integration.

---

Node-based vs Edge-based LCIA
-----------------------------

Conventional LCIA applies CFs to nodes (flows), yielding:

.. math::

   s = \mathbf{c}^T \cdot \mathbf{B} \cdot \mathbf{A}^{-1} \cdot \mathbf{f}

Where:

- :math:`\mathbf{f}` is the demand vector
- :math:`\mathbf{A}` is the technosphere matrix
- :math:`\mathbf{B}` is the biosphere matrix
- :math:`\mathbf{c}` is a vector of characterization factors
- :math:`s` is the final score

This approach treats CFs as static, context-independent.

In contrast, `edges` defines a **non-diagonal, rectangular matrix** :math:`\mathbf{C}` mapping to **individual exchanges**, enabling:

.. math::

   \mathbf{S} = \mathbf{C} \circ \mathbf{X}

Where :math:`\mathbf{X}` is the matrix of exchanges, and :math:`\mathbf{S}` is the characterized inventory.

---

Matching Logic
--------------

Characterization factors in `edges` are matched using:

- **Supplier** attributes: `name`, `reference product`, `location`, `matrix`
- **Consumer** attributes: `location`, `matrix`, `classifications`
- **Contextual parameters**: e.g., scenario-dependent variables like CO₂ concentrations

Matching is flexible: partial string matches (`contains`, `startswith`), matrix filters, nested dictionaries for `classifications`.

---

Regionalization Mechanisms
---------------------------

Regional mapping follows a five-step cascade:

1. **Direct Match**: Exact match on supplier/consumer location
2. **Disaggregation**: Split aggregate inventory region using weighted CFs
3. **Aggregation**: Fill missing CFs using containing regions
4. **Dynamic Resolution**: Handle "RoW"-like fallback with exclusion logic
5. **Global Fallback**: Default CF if no other match found

**Weights** can be based on population, GDP, water demand, etc.

---

Symbolic and Scenario-Based CFs
-------------------------------

CFs can be defined as symbolic expressions:

.. code-block:: json

    {
        "value": "GWP('CH4', H, C_CH4)"
    }

Where variables (`H`, `C_CH4`) are evaluated per scenario or year.

This enables prospective LCIA consistent with IAM or RCP scenarios.

---

Uncertainty Modeling
--------------------

`edges` supports multiple uncertainty types:

- `normal`, `lognormal`, `triangular`, `uniform`
- `discrete_empirical` (e.g. basin-based CFs)
- Nested uncertainty: e.g., a distribution selected per region, with per-basin variability

Monte Carlo draws are supported directly via CF definitions and sampling logic.

---

Technosphere CFs
----------------

Beyond biosphere flows, `edges` can characterize **technosphere exchanges** — for instance, country-to-country commodity flows (e.g., GeoPolRisk). This adds relational, supply-chain-sensitive impact modeling.

---

Life Cycle Costing (LCC)
------------------------

In addition to environmental impact assessment, the `edges` framework supports
**life cycle costing (LCC)** through the `CostLCIA` class. This allows the user
to apply cost-based characterization factors to LCI exchanges, using the same
regionalization and parametrization logic available for environmental LCIA.

Cost data can be:

- Fixed (e.g., per unit cost in a specific currency)
- Parameterized (e.g., price = base_price * inflation_factor)
- Scenario-dependent (e.g., modeled price paths from IAMs or forecasts)
- Uncertain (e.g., cost ranges or distributions used in Monte Carlo)

The core LCC workflow mirrors LCIA:

1. Compute the inventory with `lci()`
2. Match exchanges with cost factors via `map_exchanges()` (and mapping steps)
3. Evaluate symbolic costs via `evaluate_cfs()`
4. Compute the total life cycle cost using `lcia()`
5. Export detailed cost breakdowns via `generate_cf_table()`

This allows integrated techno-economic and environmental modeling under consistent spatial, structural, and scenario assumptions.

**Example:**

.. code-block:: python

    import bw2data
    from edges import CostLCIA

    bw2data.projects.set_current("some project")
    act = bw2data.Database("some db").random()

    lcia = CostLCIA(
        method=("LCC 1.0", "2023"),
        demand={act: 1}
    )

    lcia.lci()
    lcia.map_exchanges()
    lcia.evaluate_cfs()
    lcia.lcia()
    df = lcia.generate_cf_table()
    print(df.head())

---

Summary
-------

By resolving CFs at the level of exchanges, `edges`:

- Enables context-aware LCIA without full GIS dependency
- Supports regionalized, relational, and prospective modeling
- Keeps logic transparent, reproducible, and extendable via JSON and symbolic expressions

This makes `edges` ideal for advanced LCIA applications where geography, classification, and policy scenarios all matter.
