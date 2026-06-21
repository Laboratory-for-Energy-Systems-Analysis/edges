
Available Methods
=================

This section describes the built-in LCIA methods available in `edges`, including their purpose, implementation, citation, minimal example, and a JSON schema excerpt showing how exchanges are matched.

Some built-in methods, such as the IBIF ``all pressures`` variants, mix
``biosphere-technosphere`` and ``technosphere-technosphere`` CF rows in one
method file. The standard ``EdgeLCIA`` workflow is unchanged for these methods;
``generate_cf_table()`` reports the contribution family via the
``supplier matrix`` and ``direction`` columns.

---

AWARE 2.0
---------

**Name**: `AWARE 2.0` (and variants)

**Impact Category**:

- ``("AWARE 2.0", "Country", "all", "yearly")``
- ``("AWARE 2.0", "Country", "irri", "yearly")``
- ``("AWARE 2.0", "Country", "non_irri", "yearly")``
- ``("AWARE 2.0", "Country", "unspecified", "yearly")``
- ``("AWARE 2.0 prospective", "Country", "all", "yearly")``
- ``("AWARE 2.0 prospective", "Country", "irri", "yearly")``
- ``("AWARE 2.0 prospective", "Country", "non_irri", "yearly")``
- ``("AWARE 2.0 prospective", "Country", "unspecified", "yearly")``

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

    bw2data.projects.set_current("some project")
    act = bw2data.Database("some db").random()

    lcia = EdgeLCIA(
        demand={act: 1},
        method=("AWARE 2.0", "Country", "all", "yearly")
    )

    lcia.lci()

    # you can use .apply_strategies() since the strategies are listed in the LCIA JSON
    lcia.apply_strategies()
    # if not, use the following mapping methods:
    #lcia.map_exchanges() # finds direct matches
    #lcia.map_aggregate_locations() # finds matches for aggregate regions ("RER", "US" etc.)
    #lcia.map_dynamic_locations() # finds matches for dynamic regions ("RoW", "RoW", etc.)
    #lcia.map_contained_locations() # finds matches for contained regions ("CA" for "CA-QC" if factor of "CA-QC" is not available)
    #lcia.map_remaining_locations_to_global() # applies global factors to remaining locations
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

Prospective AWARE 2.0
^^^^^^^^^^^^^^^^^^^^^

The prospective variants provide country-average annual CFs for ``SSP126``,
``SSP370``, and ``SSP585`` in five-year steps from 2019 to 2049. They use the
same four scopes as the static AWARE 2.0 methods; the ``all`` variant uses CPC
classification to distinguish agricultural and non-agricultural consumers.
Rows keep numeric ``value`` and ``weight`` fields for the baseline
``SSP126``/``2019`` case, while ``value_expression`` and
``weight_expression`` select the requested scenario/year during evaluation.
This keeps the files close to the static AWARE 2.0 structure while allowing
scenario-dependent country averages and aggregate fallback weights.
Requested years between five-year data points are linearly interpolated.
Requested years outside 2019-2049 use the closest available endpoint by
default.
The method files document this policy with explicit interpolation metadata:

.. code-block:: json

    {
      "interpolation": {
        "axis": "scenario_idx",
        "axis_type": "year",
        "method": "linear",
        "extrapolation": "nearest",
        "source_years": ["2019", "2024", "2029", "2034", "2039", "2044", "2049"]
      }
    }

Evaluate a specific scenario/year with:

.. code-block:: python

    lcia = EdgeLCIA(
        demand={act: 1},
        method=("AWARE 2.0 prospective", "Country", "all", "yearly"),
        scenario="SSP585",
    )
    lcia.lci()
    lcia.apply_strategies()
    lcia.evaluate_cfs(scenario_idx="2049")
    lcia.lcia()

The prospective methods include country-level weights for aggregate fallback.
When ``use_distributions=True``, country/category CFs with basin geometry
coverage sample basin-specific CFs using basin weights allocated to countries
from the AWARE basin GeoPackage.

In deterministic mode, the exchange mapping can be performed once and the same
``EdgeLCIA`` object can be re-evaluated for each scenario/year pair. Years do
not need to be limited to the five-year source grid:

.. code-block:: python

    for scenario in ["SSP126", "SSP370", "SSP585"]:
        for year in ["2019", "2026", "2031", "2049", "2055"]:
            lcia.evaluate_cfs(scenario=scenario, scenario_idx=year)
            lcia.lcia()
            print(scenario, year, lcia.score)

For stochastic runs, pass ``use_distributions=True`` and ``iterations=...`` to
``EdgeLCIA``. Scenario/year-specific ``values_by_scenario`` and
``weights_by_scenario`` arrays are selected during ``evaluate_cfs``. The
prospective AWARE distributions include basin IDs, so intermediate years are
interpolated by aligning basin-specific values and weights before sampling.

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

    bw2data.projects.set_current("some project")
    act = bw2data.Database("some db").random()

    lcia = EdgeLCIA(
        demand={act: 1},
        method=("GeoPolRisk", "paired", "2024")
    )

    lcia.lci()
    # you can use .apply_strategies(), since the strategies are listed in the LCIA JSON
    lcia.apply_strategies()
    # if not, use the following mapping methods:
    #lcia.map_exchanges()
    #lcia.map_aggregate_locations()
    #lcia.map_contained_locations()
    #lcia.map_remaining_locations_to_global()
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

    bw2data.projects.set_current("some project")
    act = bw2data.Database("some db").random()

    lcia = EdgeLCIA(
        demand={act: 1},
        method=("ImpactWorld+ 2.1", "Freshwater acidification", "midpoint")
    )

    lcia.lci()
    # you can use .apply_strategies() since the strategies are listed in the LCIA JSON
    lcia.apply_strategies()
    # if not, use the following mapping methods:
    #lcia.map_exchanges()
    #lcia.map_aggregate_locations()
    #lcia.map_dynamic_locations()
    #lcia.map_contained_locations()
    #lcia.map_remaining_locations_to_global()
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
--------------------------------

**Name**: `SCP_1.0.json`

**Impact Category**: Fossil Fuel Resource Scarcity

**Usage Example**:

.. code-block:: python

    import bw2data
    from edges import EdgeLCIA

    bw2data.projects.set_current("some project")
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

    bw2data.projects.set_current("some project")
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

See also:

For deterministic regionalized methods, ``generate_cf_table(split_aggregate_consumers=True)``
can be used to replace weighted fallback consumer regions with country-specific
rows in the exported results table.

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

``GWP`` must be supplied as a trusted callable, for example via
``EdgeLCIA(..., allowed_functions={"GWP": GWP})``. Method expressions are
sandboxed to arithmetic, parameters, literals, and bare allowlisted function
calls; arbitrary Python syntax is not supported.

**Reference**:
IPCC AR6, 2021.
https://www.ipcc.ch/assessment-report/ar6/


---

GLAM3 - Land use impacts on biodiversity
----------------------------------------

**Name**: `Land use impacts on biodiversity`

**Impact Category**:

- ``("GLAM3", "biodiversity", "occupation", "average", "amphibians")``
- ``("GLAM3", "biodiversity", "occupation", "average", "birds")``
- ``("GLAM3", "biodiversity", "occupation", "average", "eukaryota")``
- ``("GLAM3", "biodiversity", "occupation", "average", "mammals")``
- ``("GLAM3", "biodiversity", "occupation", "average", "plants")``
- ``("GLAM3", "biodiversity", "occupation", "average", "reptiles")``
- ``("GLAM3", "biodiversity", "transformation", "average", "amphibians")``
- ``("GLAM3", "biodiversity", "transformation", "average", "birds")``
- ``("GLAM3", "biodiversity", "transformation", "average", "eukaryota")``
- ``("GLAM3", "biodiversity", "transformation", "average", "mammals")``
- ``("GLAM3", "biodiversity", "transformation", "average", "plants")``
- ``("GLAM3", "biodiversity", "transformation", "average", "reptiles")``

These methods present different scopes:

* Occupation CFs (in PDF·m²⁻¹) quantify impacts per unit area and time of land occupation.
* Transformation CFs (in PDF·yr·m²⁻¹) quantify impacts per unit area transformed, accounting for regeneration time.
* CFs are provided as average, at country level, with ecoregion-specific CFs for sensitivity purpose, across five biome types.
* The CFs integrate land fragmentation (via the Equivalent Connected Area, ECA) and land use intensity—both drivers of biodiversity loss not jointly included in earlier models.
* CFs cannot be used to assess changes in fragmentation degree, as fragmentation is internally parameterized.
* The Eukaryota CFs represent an aggregated proxy combining plants and vertebrates (amphibians, birds, mammals, reptiles), each weighted equally. They thus serve as a generic biodiversity indicator for overall potential species loss across taxa.

**Description**:

This method, developed within the UNEP Life Cycle Initiative’s GLAM3 framework,
quantifies biodiversity impacts of land use and land-use change on species
richness using updated global data. It combines two major advancements:

1. inclusion of land use intensity and
2. explicit consideration of habitat fragmentation.

Characterization factors (CFs) were derived using a countryside species–area
relationship adjusted for fragmentation, with relative species loss calculated
separately for five species groups — plants, amphibians, birds, mammals, and reptiles.
Regional CFs were scaled to global species loss using extinction probabilities.

The resulting CFs represent potential, long-term losses in species richness
(accounting for extinction debt), expressed in potentially disappeared
fraction of species (PDF). They are available for both land occupation and land
transformation, enabling integration into LCA at multiple spatial scales.

**Usage Example**:

.. code-block:: python

    import bw2data
    from edges import EdgeLCIA

    bw2data.projects.set_current("some project")
    act = bw2data.Database("some db").random()

    lcia = EdgeLCIA(
        demand={act: 1},
        method=("GLAM3", "biodiversity", "occupation", "average", "amphibians")
    )

    lcia.lci()

    # you can use .apply_strategies() since the strategies are listed in the LCIA JSON
    lcia.apply_strategies()
    # if not, use the following mapping methods:
    #lcia.map_exchanges() # finds direct matches
    #lcia.map_aggregate_locations() # finds matches for aggregate regions ("RER", "US" etc.)
    #lcia.map_dynamic_locations() # finds matches for dynamic regions ("RoW", "RoW", etc.)
    #lcia.map_contained_locations() # finds matches for contained regions ("CA" for "CA-QC" if factor of "CA-QC" is not available)
    #lcia.map_remaining_locations_to_global() # applies global factors to remaining locations
    lcia.evaluate_cfs()
    lcia.lcia()

**Sample CF JSON**:

.. code-block:: json

    {
      "supplier": {
        "name": "Occupation, annual crop, greenhouse",
        "categories": [
          "natural resource",
          "land"
        ],
        "matrix": "biosphere"
      },
      "consumer": {
        "location": "AF",
        "matrix": "technosphere"
      },
      "value": 2.99558005008489e-12,
      "weight": 643830.7383041249,
      "uncertainty": {
        "distribution": "discrete_empirical",
        "parameters": {
          "values": [
            7.79534374146956e-12,
            1.50524230465717e-11,
            3.80734278838771e-12,
            9.985618271115879e-12,
            9.985618271115879e-12,
            9.985618271115879e-12,
            9.985618271115879e-12,
            9.985618271115879e-12,
            9.985618271115879e-12,
            1.0017404981279698e-11,
            4.595297532330699e-12,
            1.6189133164701399e-12,
            3.52822420899683e-12,
            7.80635977979514e-13,
            1.0017404981279698e-11,
            1.60177437279945e-12
          ],
          "weights": [
            0.0002681212386553088,
            0.01986688334294251,
            0.005598136518128794,
            0.10351496842799959,
            0.04397028921970691,
            0.006936061677420315,
            0.0019971425916960298,
            0.007822246506266655,
            0.006322221724374011,
            0.021299462915867963,
            0.08460533036349488,
            0.05054301639029757,
            0.21687159321700622,
            0.0364686471004479,
            0.14403400105010292,
            0.24874318136298704
          ]
        }
      }
    }

**Reference**:
Scherer L, Rosa F, Sun Z, et al (2023)
Biodiversity Impact Assessment Considering Land Use Intensities and Fragmentation.
Environ Sci Technol https://doi.org/10.1021/acs.est.3c04191

---

IBIF v2 - Biodiversity intactness
---------------------------------

**Name**: `IBIF biodiversity`

**Impact Categories**:

- ``("IBIF", "biodiversity", "CO2", "overall")``
- ``("IBIF", "biodiversity", "CO2", "plants")``
- ``("IBIF", "biodiversity", "CO2", "vertebrates")``
- ``("IBIF", "biodiversity", "NH3", "overall")``
- ``("IBIF", "biodiversity", "NH3", "plants")``
- ``("IBIF", "biodiversity", "NOx", "overall")``
- ``("IBIF", "biodiversity", "NOx", "plants")``
- ``("IBIF", "biodiversity", "LU", "overall")``
- ``("IBIF", "biodiversity", "LU", "plants")``
- ``("IBIF", "biodiversity", "LU", "vertebrates")``
- ``("IBIF", "biodiversity", "roads", "overall")``
- ``("IBIF", "biodiversity", "roads", "vertebrates")``
- ``("IBIF", "biodiversity", "all pressures", "overall")``
- ``("IBIF", "biodiversity", "all pressures", "plants")``
- ``("IBIF", "biodiversity", "all pressures", "vertebrates")``

These methods present different scopes:

* ``CO2``, ``NH3``, and ``NOx`` are biosphere-to-technosphere emission methods.
* ``LU`` is a biosphere-to-technosphere land occupation method.
* ``roads`` is a technosphere-to-technosphere method targeting road
  infrastructure suppliers with unit ``meter-year``.
* ``all pressures`` combines the relevant pressure families into one built-in
  method. ``overall`` and ``vertebrates`` now include road pressure rows,
  while ``plants`` remains biosphere-only because the source IBIF release does
  not provide a road CF column for plants.

**Description**:

IBIF v2 provides country-level biodiversity-intactness factors derived from the
GLOBIO 4 model and expressed in mean species abundance (MSA) loss space.
Depending on the pressure family, the built-in ``edges`` methods characterize
inventory emissions, land occupation, or road infrastructure. The dedicated
``roads`` methods convert published road-length CFs from per-kilometer to
per-meter so they can match ecoinvent ``meter-year`` road infrastructure
suppliers. The mixed ``all pressures`` methods combine emissions, land use, and
roads in one JSON file.

**Usage Example**:

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
    print(df[["supplier matrix", "direction", "amount", "CF", "impact"]].head())

**Sample CF JSON**:

.. code-block:: json

    {
      "supplier": {
        "name": "road",
        "operator": "contains",
        "excludes": ["market", "maintenance", "treatment"],
        "unit": "meter-year",
        "location": "AF",
        "matrix": "technosphere"
      },
      "consumer": {
        "matrix": "technosphere"
      },
      "value": 0.000392,
      "weight": 17152234636.8715
    }

This sample row comes from ``("IBIF", "biodiversity", "roads", "overall")``.
The corresponding ``all pressures`` methods append these road rows to the
emission and land occupation CF rows already present in the biosphere side of
the method.

**Reference**:
Schipper, A. M., et al. (2025)
Intactness-Based Biodiversity Impact Factors (IBIF), version 2.
Scientific Data.
https://doi.org/10.1038/s41597-025-05946-1
