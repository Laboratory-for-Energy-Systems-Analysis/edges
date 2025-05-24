
Available Methods
=================

This section describes the built-in LCIA methods available in `edges`, including their purpose, implementation, citation, minimal example, and a JSON schema excerpt showing how exchanges are matched.

---

AWARE 2.0
---------

**Name**: `AWARE 2.0` (and variants)

**Impact Category**:

- ``("AWARE 2.0", "Country", "all", "yearly")``
- ``("AWARE 2.0", "Country", "irri", "yearly")``
- ``("AWARE 2.0", "Country", "non_irri", "yearly")``
- ``("AWARE 2.0", "Country", "unspecified", "yearly")``

These four methods present different scopes:

- ``all``: applies consumption type-specific CFs depending on the agricultural, non-agrilcultural and unspecified nature of the consumer. Uses CPC codes.
- ``irri``: applies CFs considering that all consumers are agricultural activities. Uses CPC codes.
- ``non_irri``: applies CF considering that all consumers are non-agricultural activities. Uses CPC codes.
- ``unspecified``: applies CF to all consumers, without distinction based on consumption pattern. Does not use CPC codes.


**Description**: AWARE estimates water deprivation potential by measuring the availability of water after human and ecosystem needs are met.

**Usage Example**:

.. code-block:: python

    import bw2data
    from edges import EdgeLCIA

    bw2data.project.set_current("some project")
    act = bw2data.Database("some db").random()

    lcia = EdgeLCIA(
        demand={act: 1},
        method=("AWARE 2.0", "Country", "all", "yearly")
    )

    lcia.lci()
    lcia.map_exchanges() # finds direct matches
    lcia.map_aggregate_locations() # finds matches for aggregate regions ("RER", "US" etc.)
    lcia.map_dynamic_locations() # finds matches for dynamic regions ("RoW", "RoW", etc.)
    lcia.map_contained_locations() # finds matches for contained regions ("CA" for "CA-QC" if factor of "CA-QC" is not available)
    lcia.map_remaining_locations_to_global() # applies global factors to remaining locations
    lcia.evaluate_cfs()
    lcia.lcia()

**Sample CF JSON**:

.. code-block:: json

    {
      "supplier": {
        "name": "Water, lake",
        "categories": ["natural resource", "in water"],
        "matrix": "biosphere"
      },
      "consumer": {
        "location": "AM",
        "matrix": "technosphere",
        "classifications": {"CPC": ["01"]}
      },
      "value": 88.6,
      "weight": 799882000,
      "uncertainty": {
        "distribution": "discrete_empirical",
        "parameters": {
          "values": [84.5, 87.9],
          "weights": [0.031, 0.969]
        }
      }
    }

Here `"classifications": {"CPC": ["01"]}` ensures that this CF only applies
to agriclutural processes.

**Reference**:
Seitfudem, G., Berger, M., Schmied, H. M., & Boulay, A.-M. (2025).
The updated and improved method for water scarcity impact assessment in LCA, AWARE2.0.
Journal of Industrial Ecology, 1–17.
https://doi.org/10.1111/jiec.70023

---

GeoPolRisk 1.0
--------------

**Name**: `GeoPolRisk_2024.json`

**Impact Category**:

- ``("GeoPolRisk", "paired", "2024")``
- ``("GeoPolRisk", "2024")``

``("GeoPolRisk", "2024")`` applies factors solely based on the metal consumer's location.
``("GeoPolRisk", "paired", "2024")`` applies factors based on supplying-consuming location pairs.

**Usage Example**:

.. code-block:: python

    import bw2data
    from edges import EdgeLCIA

    bw2data.project.set_current("some project")
    act = bw2data.Database("some db").random()

    lcia = EdgeLCIA(
        demand={act: 1},
        method=("GeoPolRisk", "paired", "2024")
    )

    lcia.lci()
    lcia.map_exchanges()
    lcia.map_aggregate_locations()
    lcia.map_contained_locations()
    lcia.map_remaining_locations_to_global()
    lcia.evaluate_cfs()
    lcia.lcia()

**Sample CF JSON**:

.. code-block:: json

    {
      "supplier": {
        "name": "aluminium production",
        "reference product": "aluminium",
        "location": "AU",
        "operator": "startswith",
        "matrix": "technosphere"
      },
      "consumer": {
        "location": "CA",
        "matrix": "technosphere"
      },
      "value": 1.10e-10
    }

**Reference**:  
Anish Koyamparambath, Philippe Loubet, Steven B. Young, Guido Sonnemann (2024)
Spatially and temporally differentiated characterization factors for supply risk of abiotic resources in life cycle assessment,
Resources, Conservation and Recycling,
https://doi.org/10.1016/j.resconrec.2024.107801.

---

ImpactWorld+ 2.1
----------------

**Name**: `ImpactWorld+ 2.1_<category>_<level>.json`

**Impact Categories**:

- ``("ImpactWorld+ 2.1", "Freshwater acidification", "damage")``
- ``("ImpactWorld+ 2.1", "Freshwater acidification", "midpoint")``
- ``("ImpactWorld+ 2.1", "Freshwater ecotoxicity, long term", "damage")``
- ``("ImpactWorld+ 2.1", "Freshwater ecotoxicity, long term", "midpoint")``
- ``("ImpactWorld+ 2.1", "Freshwater ecotoxicity, short term", "damage")``
- ``("ImpactWorld+ 2.1", "Freshwater ecotoxicity, short term", "midpoint")``
- ``("ImpactWorld+ 2.1", "Freshwater ecotoxicity", "damage")``
- ``("ImpactWorld+ 2.1", "Freshwater ecotoxicity", "midpoint")``
- ``("ImpactWorld+ 2.1", "Freshwater eutrophication", "damage")``
- ``("ImpactWorld+ 2.1", "Freshwater eutrophication", "midpoint")``
- ``("ImpactWorld+ 2.1", "Land occupation, biodiversity", "damage")``
- ``("ImpactWorld+ 2.1", "Land occupation, biodiversity", "midpoint")``
- ``("ImpactWorld+ 2.1", "Land transformation, biodiversity", "damage")``
- ``("ImpactWorld+ 2.1", "Land transformation, biodiversity", "midpoint")``
- ``("ImpactWorld+ 2.1", "Marine ecotoxicity, long term", "damage")``
- ``("ImpactWorld+ 2.1", "Marine ecotoxicity, long term", "midpoint")``
- ``("ImpactWorld+ 2.1", "Marine ecotoxicity, short term", "damage")``
- ``("ImpactWorld+ 2.1", "Marine ecotoxicity, short term", "midpoint")``
- ``("ImpactWorld+ 2.1", "Marine eutrophication", "damage")``
- ``("ImpactWorld+ 2.1", "Marine eutrophication", "midpoint")``
- ``("ImpactWorld+ 2.1", "Particulate matter formation", "damage")``
- ``("ImpactWorld+ 2.1", "Particulate matter formation", "midpoint")``
- ``("ImpactWorld+ 2.1", "Photochemical ozone formation, ecosystem quality", "damage")``
- ``("ImpactWorld+ 2.1", "Photochemical ozone formation, ecosystem quality", "midpoint")``
- ``("ImpactWorld+ 2.1", "Photochemical ozone formation, human health", "damage")``
- ``("ImpactWorld+ 2.1", "Photochemical ozone formation, human health", "midpoint")``
- ``("ImpactWorld+ 2.1", "Photochemical ozone formation", "damage")``
- ``("ImpactWorld+ 2.1", "Photochemical ozone formation", "midpoint")``
- ``("ImpactWorld+ 2.1", "Terrestrial acidification", "damage")``
- ``("ImpactWorld+ 2.1", "Terrestrial acidification", "midpoint")``
- ``("ImpactWorld+ 2.1", "Terrestrial ecotoxicity, long term", "damage")``
- ``("ImpactWorld+ 2.1", "Terrestrial ecotoxicity, long term", "midpoint")``
- ``("ImpactWorld+ 2.1", "Terrestrial ecotoxicity, short term", "damage")``
- ``("ImpactWorld+ 2.1", "Terrestrial ecotoxicity, short term", "midpoint")``
- ``("ImpactWorld+ 2.1", "Thermally polluted water", "damage")``
- ``("ImpactWorld+ 2.1", "Thermally polluted water", "midpoint")``
- ``("ImpactWorld+ 2.1", "Water availability, freshwater ecosystem", "damage")``
- ``("ImpactWorld+ 2.1", "Water availability, freshwater ecosystem", "midpoint")``
- ``("ImpactWorld+ 2.1", "Water availability, human health", "damage")``
- ``("ImpactWorld+ 2.1", "Water availability, human health", "midpoint")``
- ``("ImpactWorld+ 2.1", "Water availability, terrestrial ecosystem", "damage")``
- ``("ImpactWorld+ 2.1", "Water availability, terrestrial ecosystem", "midpoint")``
- ``("ImpactWorld+ 2.1", "Water scarcity", "damage")``
- ``("ImpactWorld+ 2.1", "Water scarcity", "midpoint")``



**Usage Example**:

.. code-block:: python

    import bw2data
    from edges import EdgeLCIA

    bw2data.project.set_current("some project")
    act = bw2data.Database("some db").random()

    lcia = EdgeLCIA(
        demand={act: 1},
        method=("ImpactWorld+ 2.1", "Freshwater acidification", "midpoint")
    )

    lcia.lci()
    lcia.map_exchanges()
    lcia.map_aggregate_locations()
    lcia.map_dynamic_locations()
    lcia.map_contained_locations()
    lcia.map_remaining_locations_to_global()
    lcia.evaluate_cfs()
    lcia.lcia()

**Sample CF JSON**:

.. code-block:: json

    {
      "supplier": {
        "name": "Ammonia",
        "categories": [
          "air"
        ],
        "matrix": "biosphere"
      },
      "consumer": {
        "location": "AD",
        "matrix": "technosphere"
      },
      "value": 0.1801410433590999
    }

**Reference**:  
Bulle, C., Margni, M., Patouillard, L. et al.
IMPACT World+: a globally regionalized life cycle impact assessment method.
Int J Life Cycle Assess 24, 1653–1674 (2019).
https://doi.org/10.1007/s11367-019-01583-0

---

SCP 1.0 (Surplus Cost Potential)
-------------------------------

**Name**: `SCP_1.0.json`

**Impact Category**: Fossil Fuel Resource Scarcity

**Usage Example**:

.. code-block:: python

    import bw2data
    from edges import EdgeLCIA

    bw2data.project.set_current("some project")
    act = bw2data.Database("some db").random()

    lcia = EdgeLCIA(
        demand={act: 1},
        method=("SCP", "1.0")
    )

    lcia.lci()
    lcia.map_exchanges()
    lcia.evaluate_cfs(parameters={"MCI_OIL": 0.5, "P_OIL": 400, "d": 0.03})
    lcia.lcia()

**Sample CF JSON**:

.. code-block:: json

    {
      "supplier": {
        "name": "Oil, crude",
        "categories": ["natural resource", "in ground"],
        "matrix": "biosphere"
      },
      "consumer": {
        "matrix": "technosphere"
      },
      "value": "(MCI_OIL * P_OIL / 5) / (1 + d)"
    }

**Reference**:  
Loosely adapted from:

Vieira, M.D.M., Huijbregts, M.A.J.
Comparing mineral and fossil surplus costs of renewable and non-renewable electricity production.
Int J Life Cycle Assess 23, 840–850 (2018).
https://doi.org/10.1007/s11367-017-1335-6

---

Parameterized GWP
-----------------

**Name**: `lcia_parameterized_gwp.json`

**Impact Category**: Global Warming Potential (Dynamic)

**Usage Example**:

.. code-block:: python

    import bw2data
    from edges import EdgeLCIA

    bw2data.project.set_current("some project")
    act = bw2data.Database("some db").random()

    # Define scenario parameters (e.g., atmospheric CO₂ concentration and time horizon)
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

See also:

- examples/simple_parameterized_example_1.json

**Sample CF JSON**:

.. code-block:: json

    {
      "supplier": {
        "name": "Methane, fossil",
        "matrix": "biosphere",
        "operator": "contains"
      },
      "consumer": {
        "matrix": "technosphere"
      },
      "value": "GWP('CH4', H, C_CH4)"
    }

**Reference**:
IPCC AR6, 2021.
https://www.ipcc.ch/assessment-report/ar6/
