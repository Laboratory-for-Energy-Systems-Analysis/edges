
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

    bw2data.project.set_current("some project")
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
    df = lcia.generate_df_table()
    print(df.head())

---

Using Built-in Method Files
---------------------------

You can list available method files with:

.. code-block:: python

    from edges import get_available_methods
    print(get_available_methods())

Use the name in the `method=` argument when instantiating `EdgeLCIA`.

---

Regionalized LCIA
-----------------

When using region-specific methods like AWARE or ImpactWorld+:

.. code-block:: python

    import bw2data
    from edges import EdgeLCIA

    bw2data.project.set_current("some project")
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

---



Using a Custom Method JSON
--------------------------

Your method file should follow the expected CF JSON schema:

.. code-block:: python

    import bw2data
    from edges import EdgeLCIA

    bw2data.project.set_current("some project")
    act = bw2data.Database("some db").random()

    lcia = EdgeLCIA(method="my_custom_method.json")
    lcia.lci()
    lcia.map_exchanges()
    lcia.evaluate_cfs(parameters={"H": 100, "C_CH4": 1866})
    lcia.lcia()

---

Parameterized CFs
-----------------

If the method uses symbolic expressions, pass parameter values:

.. code-block:: python

    import bw2data
    from edges import EdgeLCIA

    bw2data.project.set_current("some project")
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
        filepath="lcia_parameterized_gwp.json")
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

---

Uncertainty-aware LCIA
-----------------------

If CFs include uncertainty (e.g., lognormal, discrete empirical),
you can get statistics:

.. code-block:: python

    import bw2data
    from edges import EdgeLCIA

    bw2data.project.set_current("some project")
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
    import matplitlib.pyplot as plt
    plt.hist(lcia.score, bins=100)

    # get dataframe with statistics
    df = lcia.generate_cf_table()


---

To know more on how uncertainty works in `edges`, see:

- examples/uncertainty.ipynb

Working with Technosphere CFs (e.g., GeoPolRisk)
------------------------------------------------

.. code-block:: python

    import bw2data
    from edges import EdgeLCIA

    bw2data.project.set_current("some project")
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
    df = lcia.generate_df_table()
    df.to_csv("results.csv")

---

Scenario-based Fossil Resource Scarcity
---------------------------------------

Supports expressions depending on extraction volume and discount rate:

.. code-block:: python

    import bw2data
    from edges import EdgeLCIA

    bw2data.project.set_current("some project")
    act = bw2data.Database("some db").random()

    lcia = EdgeLCIA(method="SCP_1.0.json")
    lcia.lci()
    lcia.map_exchanges()
    lcia.evaluate_cfs(parameters={"MCI_OIL": 0.5, "P_OIL": 450, "d": 0.03})
    lcia.lcia()

---

Exporting Results
-----------------

You can inspect or save the detailed contribution table:

.. code-block:: python

    df = lcia.generate_cf_table()
    df.to_csv("edge_lcia_detailed_results.csv")
