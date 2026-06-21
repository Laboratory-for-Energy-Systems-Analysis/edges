
User Guide
==========

This guide walks you through practical usage patterns of the `edges` library,
covering common workflows such as simple LCIA, regionalized impact assessment,
parameterized methods, uncertainty analysis, and scenario-based modeling.

---

Simple LCIA
-----------

For non-regionalized methods with fixed CFs:

.. code-block:: python

    import bw2data
    from edges import EdgeLCIA

    bw2data.projects.set_current("some project")
    act = bw2data.Database("some db").random()

    # here, the user provides his/her own LCIA method file
    lcia = EdgeLCIA(
        demand={act: 1},
        filepath="lcia_example_1.json"
    )
    # solves the system and generates the inventory matrix
    lcia.lci()
    # map exchanges that should receive a CF
    lcia.map_exchanges()
    # evaluate the CF values
    lcia.evaluate_cfs()
    # populate the characterized_inventory matrix and a score
    lcia.lcia()
    print(lcia.score)
    # optional but RECOMMENDED, generate a dataframe with all characterized exchanges
    # this allows you to check whether exchanges have been given the correct CFs
    # include_unmatched=True allows you to see which exchanges were not matched (and if some should have been)
    df = lcia.generate_cf_table()
    print(df.head())

---

Using Built-in Method Files
---------------------------

You can list available method files with:

.. code-block:: python

    from edges import get_available_methods
    print(get_available_methods())

Use the name in the `method=` argument when instantiating `EdgeLCIA`.

.. note::
   The matcher backend is `CLIPSpy <https://clipspy.readthedocs.io/en/latest/>`_
   (Python wrapper for `CLIPS <http://www.clipsrules.net/>`_). Use the default
   ``matcher_backend="clips"``.

---

Regionalized LCIA
-----------------

When using region-specific methods like AWARE or ImpactWorld+:

.. code-block:: python

    import bw2data
    from edges import EdgeLCIA

    bw2data.projects.set_current("some project")
    act = bw2data.Database("some db").random()

    # here, we use a method already included in `edges`
    lcia = EdgeLCIA(
        demand={act: 1},
        method=("AWARE 2.0", "Country", "all", "yearly"),
    )
    lcia.lci()
    lcia.map_exchanges()
    # this is a regionalized LCIa method
    # so a few extra steps are necessary to ensure
    # that exchanges with suppliers located in aggregated regions (e.g., RER)
    # or dynamic regions (e.g., RoW) also get a CF
    lcia.map_aggregate_locations()
    lcia.map_dynamic_locations()
    lcia.map_contained_locations()
    lcia.map_remaining_locations_to_global()
    lcia.evaluate_cfs()
    lcia.lcia()
    print(lcia.score)

For deterministic regionalized runs, ``generate_cf_table(split_aggregate_consumers=True)``
replaces weighted fallback rows for aggregate or dynamic consumer regions with
country-specific rows in the exported table.

.. note::
   Methods can mix both ``supplier.matrix = "biosphere"`` and
   ``supplier.matrix = "technosphere"`` entries in the same JSON. ``edges``
   builds both edge families during ``lci()``, sums both contributions in
   ``lcia()``, and exposes them in ``generate_cf_table()`` via the
   ``supplier matrix`` and ``direction`` columns.

---



Using a Custom Method JSON
--------------------------

Your method file should follow the expected CF JSON schema:

Supplier and consumer matching keys can include ``unit`` when otherwise identical
flows need to remain distinct during CF matching.

.. code-block:: python

    import bw2data
    from edges import EdgeLCIA

    bw2data.projects.set_current("some project")
    act = bw2data.Database("some db").random()

    lcia = EdgeLCIA(
        demand={act: 1},
        method="my_custom_method.json",
    )
    lcia.lci()
    lcia.map_exchanges()
    lcia.evaluate_cfs(parameters={"H": 100, "C_CH4": 1866})
    lcia.lcia()

---

Parameterized CFs
-----------------

If the method uses symbolic expressions, pass parameter values:
Expressions support arithmetic, parameter names, literal values, and bare
allowlisted function calls such as ``GWP(...)``. They do not support arbitrary
Python syntax such as attribute access, subscripting, comprehensions, imports,
or method calls. Any Python callable supplied through ``allowed_functions`` is
trusted code and should come from the user or another trusted source.

.. code-block:: python

    import bw2data
    from edges import EdgeLCIA

    bw2data.projects.set_current("some project")
    act = bw2data.Database("some db").random()

    # Define scenario parameters (e.g., atmospheric CO₂ concentration
    # and time horizon)
    params = {
        "some scenario": {
             "co2ppm": {
                "2020": 410,
                "2050": 450,
                "2100": 500
             },
             "h": {
                "2020": 100,
                "2050": 100,
                "2100": 100
             }
        }
    }

    # Define an LCIA method name (the content will be taken from the JSON file)
    method = ('GWP', 'scenario-dependent', '100 years')

    lcia = EdgeLCIA(
        demand={act: 1},
        method=method,
        parameters=params,
        filepath="lcia_parameterized_gwp.json",
    )
    lcia.lci()
    lcia.map_exchanges()

    # Run scenarios efficiently
    results = []
    for idx in {"2020", "2050", "2100"}:
        lcia.evaluate_cfs(idx)
        lcia.lcia()
        df = lcia.generate_cf_table()

        scenario_result = {
            "scenario": idx,
            "co2ppm": params["some scenario"]["co2ppm"][idx],
            "score": lcia.score,
            "CF_table": df
        }
        results.append(scenario_result)

        print(f"Scenario (CO₂ {params['some scenario']['co2ppm'][idx]} ppm): Impact = {lcia.score}")


This allows integration with scenario data (e.g., from RCPs or IAMs).

Scenario-dependent method rows can either store expressions directly in
``value`` or keep a numeric baseline in ``value`` and put the dynamic expression
in ``value_expression``. The latter is useful for prospective regionalized
methods because the baseline row remains readable while
``evaluate_cfs(scenario=..., scenario_idx=...)`` selects the requested
scenario/year value. If aggregate fallback rows include ``weight_expression``,
their country shares are recalculated from the evaluated weights.

---

Uncertainty-aware LCIA
-----------------------

If CFs include uncertainty (e.g., lognormal, discrete empirical),
you can get statistics. In this mode, ``use_distributions=True`` samples
the characterization factors only; the Brightway inventory remains fixed:

.. code-block:: python

    import bw2data
    from edges import EdgeLCIA

    bw2data.projects.set_current("some project")
    act = bw2data.Database("some db").random()

    lcia = EdgeLCIA(
        demand={act: 1},
        method=("AWARE 2.0", "Country", "all", "yearly"),
        use_distributions=True,
        iterations=10_000
    )

    lcia.lci()
    lcia.map_exchanges()
    lcia.map_aggregate_locations()
    lcia.map_dynamic_locations()
    lcia.map_contained_locations()
    lcia.map_remaining_locations_to_global()
    lcia.evaluate_cfs()
    lcia.lcia()

    print(lcia.score.mean())

    #plot histogram of results distirbution
    import matplotlib.pyplot as plt
    plt.hist(lcia.score, bins=100)

    # get dataframe with statistics
    df = lcia.generate_cf_table()


---

Joint inventory + characterization Monte Carlo
----------------------------------------------

To propagate uncertainty from both the LCIA method and the inventory,
combine ``use_distributions=True`` with ``inventory_use_distributions=True``.
This reuses Brightway's stochastic inventory workflow and evaluates one score
per joint inventory + CF draw:

.. code-block:: python

    import bw2data
    from edges import EdgeLCIA

    bw2data.projects.set_current("some project")
    act = bw2data.Database("some db").random()

    lcia = EdgeLCIA(
        demand={act: 1},
        method=("AWARE 2.0", "Country", "all", "yearly"),
        use_distributions=True,
        inventory_use_distributions=True,
        store_inventory_samples=True,
        iterations=1_000,
    )

    lcia.lci()
    lcia.map_exchanges()
    lcia.map_aggregate_locations()
    lcia.map_dynamic_locations()
    lcia.map_contained_locations()
    lcia.map_remaining_locations_to_global()
    lcia.evaluate_cfs()
    lcia.lcia()

    # one score per joint Monte Carlo iteration
    print(lcia.score.mean())

    # with store_inventory_samples=True, amount statistics are also available
    df = lcia.generate_cf_table()
    print(df[["amount (mean)", "CF (mean)", "impact (mean)"]].head())

.. note::

   ``use_distributions=True`` on its own keeps the current ``edges`` behavior
   of varying only the characterization factors. Add
   ``inventory_use_distributions=True`` only when you want a joint Monte Carlo.

.. note::

   ``store_inventory_samples=False`` by default, to avoid storing a potentially
   large 3D inventory tensor. Without stored inventory samples, the score vector
   is still available, but ``generate_cf_table()`` cannot report per-iteration
   amount statistics for the joint run.

.. note::

   ``split_aggregate_consumers=True`` is only available for deterministic
   tables. It is not supported in uncertainty mode.

.. note::

   If you pass your own ``bw2calc.LCA`` object via ``lca=``, initialize it with
   ``use_distributions=True`` before using
   ``inventory_use_distributions=True`` in ``EdgeLCIA``.

.. note::

   Joint Monte Carlo is slower than CF-only uncertainty, because the inventory
   must be re-sampled and re-assessed at every iteration.

---

To know more on how uncertainty works in `edges`, see:

- examples/uncertainty.ipynb

Working with Technosphere CFs (e.g., GeoPolRisk)
------------------------------------------------

.. code-block:: python

    import bw2data
    from edges import EdgeLCIA

    bw2data.projects.set_current("some project")
    act = bw2data.Database("some db").random()

    lcia = EdgeLCIA(
        demand={act:1},
        method=("GeoPolRisk", "paired", "2024")
    )

    lcia.lci()
    lcia.map_exchanges()
    lcia.map_aggregate_locations()
    lcia.map_contained_locations()
    lcia.map_remaining_locations_to_global()
    lcia.evaluate_cfs()
    lcia.lcia()
    df = lcia.generate_cf_table(split_aggregate_consumers=True)
    df.to_csv("results.csv")

---

Working with Mixed Supplier Methods (e.g., IBIF all pressures)
--------------------------------------------------------------

Mixed methods combine biosphere-supplier rows and technosphere-supplier rows in
one method file. The standard workflow is unchanged:

.. code-block:: python

    import bw2data
    from edges import EdgeLCIA

    bw2data.projects.set_current("some project")
    act = bw2data.Database("some db").random()

    lcia = EdgeLCIA(
        demand={act: 1},
        method=("IBIF", "biodiversity", "all pressures", "overall")
    )

    lcia.lci()
    lcia.apply_strategies()
    lcia.evaluate_cfs()
    lcia.lcia()
    df = lcia.generate_cf_table(split_aggregate_consumers=True)

In the exported table, ``supplier matrix`` tells you whether a row came from
the biosphere or technosphere side, and ``direction`` distinguishes
``biosphere-technosphere`` from ``technosphere-technosphere`` matches.

The built-in IBIF ``all pressures`` methods now use this mixed format:

- ``("IBIF", "biodiversity", "all pressures", "overall")`` combines
  emissions, land occupation, and road infrastructure pressure.
- ``("IBIF", "biodiversity", "all pressures", "vertebrates")`` combines
  emissions, land occupation, and road infrastructure pressure for the
  vertebrate scope.
- ``("IBIF", "biodiversity", "all pressures", "plants")`` remains biosphere-only,
  because the source IBIF release does not provide a road CF column for plants.

.. note::

   Most core workflows support mixed methods, including ``lci()``,
   ``evaluate_cfs()``, ``lcia()``, ``redo_lcia()``, and
   ``generate_cf_table()``. Some higher-level analysis helpers still assume a
   single supplier matrix and may raise ``NotImplementedError``.

---

Scenario-based Fossil Resource Scarcity
---------------------------------------

Supports expressions depending on extraction volume and discount rate:

.. code-block:: python

    import bw2data
    from edges import EdgeLCIA

    bw2data.projects.set_current("some project")
    act = bw2data.Database("some db").random()

    lcia = EdgeLCIA(
        demand={act: 1},
        method="SCP_1.0.json",
    )
    lcia.lci()
    lcia.map_exchanges()
    lcia.evaluate_cfs(parameters={"MCI_OIL": 0.5, "P_OIL": 450, "d": 0.03})
    lcia.lcia()

---

Exporting Results
-----------------

You can inspect or save the detailed contribution table:

.. code-block:: python

    df = lcia.generate_cf_table(split_aggregate_consumers=True)
    df.to_csv("edge_lcia_detailed_results.csv")

For deterministic runs, ``split_aggregate_consumers=True`` replaces weighted
fallback rows with country-level consumer rows using the exact shares stored
during geographic fallback matching.

Direct matches are left unchanged. The option only expands weighted fallback
rows created during geographic fallback mapping.

For mixed methods, the exported table can contain both biosphere and
technosphere supplier rows. Use the ``supplier matrix`` and ``direction``
columns to filter the table by contribution family.

If you want to inspect the raw split for a given exchange instead of only the
expanded table, look at ``reporting_split`` on deterministic
``lcia.scenario_cfs`` entries after ``evaluate_cfs()``:

.. code-block:: python

    for cf in lcia.scenario_cfs:
        if cf.get("reporting_split"):
            print(cf["positions"], cf["consumer"].get("location"))
            print(cf["reporting_split"])
