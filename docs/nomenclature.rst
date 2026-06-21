Nomenclature and Custom LCIA Method Format
==========================================

Users can create and load their own LCIA methods into ``edges`` using
a JSON file format. This format defines characterization factors (CFs)
between *suppliers* (flows) and *consumers* (activities), supporting
regionalized, parameterized, and dynamic impact assessments.

File Structure
--------------

A custom LCIA method JSON file is structured as follows:

.. code-block:: json

    {
      "name": "Method name",
      "version": "1.0",
      "unit": "unit of the impact score",
      "description": "optional text",
      "exchanges": [ ... list of CFs ... ]
    }

Each **exchange** defines a characterization factor applied to a flow
from a **supplier** (typically a biosphere flow or technosphere process)
to a **consumer** (typically a technosphere process).

Exchange Structure
------------------

Each item in the ``exchanges`` list is a dictionary with:

- ``supplier``: Criteria to match the source flow
- ``consumer``: Criteria to match the target activity
- ``value``: The CF value (numeric or expression)
- ``weight``: Optional numeric weight (used in regional aggregation)

Supplier Fields
^^^^^^^^^^^^^^^

The supplier block may contain:

- ``name`` *(str)*: Name of the flow or activity (can use matching operators)
- ``reference product`` *(str)*: For technosphere flows, specifies the product output
- ``categories`` *(list)*: List of categories for biosphere flows
- ``location`` *(str)*: ISO country code or region
- ``matrix`` *(str)*: Required; either ``biosphere`` or ``technosphere``
- ``operator`` *(str)*: Optional; one of:

  - ``equals`` (default)
  - ``startswith``
  - ``contains``

- ``excludes`` *(list of str)*: Optional; substrings that disqualify a match

Consumer Fields
^^^^^^^^^^^^^^^

The consumer block may include:

- ``location`` *(str)*: ISO code or region name
- ``classifications`` *(dict)*: Classification codes, e.g., CPC sector codes:

  .. code-block:: json

      "classifications": {
          "CPC": ["01", "011"]
      }

- ``type`` *(str)*: Optional type hint for matching, e.g., ``process``
- ``matrix`` *(str)*: Required; should be ``technosphere``

Value Field
^^^^^^^^^^^

The ``value`` field can be:

- A **numeric value** (float or int):

  .. code-block:: json

      "value": 1.3204

- A **string expression** using parameters:

  .. code-block:: json

      "value": "28 * (1 + 0.001 * (co2ppm - 410))"

- A **function call** (if supported by the runtime):

  .. code-block:: json

      "value": "GWP('CH4', H, C_CH4)"

  These can refer to dynamic variables or scenario-dependent parameters defined in the LCIA model context.
  Function calls must use a bare function name that is explicitly available to
  the evaluator, for example through ``allowed_functions``.

String expressions are intentionally limited to arithmetic, parameter names,
literal numbers/strings/lists/tuples, and bare allowlisted function calls.
Object attribute access, subscripting, comprehensions, lambdas, imports, and
method calls are rejected. Python callables passed through ``allowed_functions``
are trusted code; the expression sandbox restricts the method expression, not
the implementation of those callables.

The optional ``value_expression`` field can be used together with a numeric
``value`` for scenario-dependent methods. ``value`` remains the numeric
baseline value stored in the method row; ``value_expression`` is evaluated at CF
evaluation time when parameters or scenarios are selected. If
``value_expression`` is absent, ``value`` is used as before, so older methods
with string expressions directly in ``value`` remain valid.

This pattern is useful for prospective methods where the JSON row should remain
readable as a baseline method, while still selecting scenario/year-specific
parameters during evaluation:

.. code-block:: json

    {
      "value": -80.5,
      "value_expression": "-cf_irri_ad"
    }

When parameter mappings use numeric keys such as years, Edges always supports
exact lookups. Linear interpolation for missing intermediate years and
nearest-year clamping outside the available range are enabled only when the
method declares the supported ``interpolation`` metadata policy:

.. code-block:: json

    {
      "interpolation": {
        "axis": "scenario_idx",
        "axis_type": "year",
        "method": "linear",
        "extrapolation": "nearest"
      }
    }

With this policy, a request for ``2026`` will interpolate between ``2024`` and
``2029``; a request for ``2055`` will use the closest available endpoint, e.g.
``2049``. Without this policy, missing indices keep the legacy fallback to the
last value in the mapping.

Weight Field
^^^^^^^^^^^^

The optional ``weight`` field is used when multiple CFs are available for a supplier-consumer pair (e.g., in regionalized methods). Weights are normalized and used to compute a weighted average.

The optional ``weight_expression`` field can be used together with ``weight`` for
scenario-dependent methods. ``weight`` remains the numeric baseline used for
geographic availability and ordering; ``weight_expression`` is evaluated at CF
evaluation time when aggregate fallback CFs are built from country-level rows.
When dynamic weights are present in an aggregate fallback split, Edges
recomputes the split shares from the evaluated weights before calculating the
aggregate CF.

Uncertainty References
^^^^^^^^^^^^^^^^^^^^^^

Large methods can define shared top-level uncertainty distributions under
``uncertainties`` and reference them from exchange rows with
``uncertainty_ref``. This avoids duplicating the same distribution for multiple
matching-equivalent water flow signatures. If an exchange needs the sampled
distribution sign flipped, set ``uncertainty_negative`` to ``1``.

For ``discrete_empirical`` uncertainties, distributions can be scenario- and
year-specific. ``ids_by_scenario`` is optional, but recommended when the values
represent named alternatives such as watershed/basin IDs because it lets Edges
align entries before interpolating arrays:

.. code-block:: json

    {
      "distribution": "discrete_empirical",
      "parameters": {
        "values_by_scenario": {
          "SSP585": {
            "2049": [12.0, 20.0, 35.0]
          }
        },
        "weights_by_scenario": {
          "SSP585": {
            "2049": [0.2, 0.5, 0.3]
          }
        },
        "ids_by_scenario": {
          "SSP585": {
            "2049": [101, 102, 103]
          }
        }
      }
    }

These arrays are selected by ``evaluate_cfs(scenario=..., scenario_idx=...)``.
If the requested scenario is absent, the sampler currently falls back to the
first scenario in the uncertainty definition. If the requested numeric
year/index is absent and the method declares the supported ``interpolation``
policy, arrays are linearly interpolated between bracketing years or clamped to
the closest available endpoint outside the available range. Without that
policy, missing indices keep the legacy fallback to the last value in the
scenario mapping. Generated method files should therefore keep scenario labels
consistent with the intended evaluation calls.

Matching Logic
--------------

Matching is performed between exchange definitions and actual flows/activities in the LCI database. Matching criteria include:

- Exact or pattern-based name/reference matching (`equals`, `startswith`, `contains`)
- Location matching with fallback to broader regions (e.g., from IT to RER to GLO)
- Classification matching (e.g., CPC sectors)
- Optional excludes for disqualifying candidates

Advanced Features
-----------------

Parameterized Values
^^^^^^^^^^^^^^^^^^^^

Values can include parameters such as ``co2ppm``, ``year``, or scenario-specific variables. These are evaluated at runtime using the current parameter values.

.. code-block:: json

    "value": "265 * (1 + 0.0005 * (co2ppm - 410))"

Classification Matching
^^^^^^^^^^^^^^^^^^^^^^^

Classifications are matched hierarchically. For example, if the CF uses:

.. code-block:: json

    "classifications": {
        "CPC": ["01"]
    }

Then it will match any activity whose CPC classification starts with ``01``, such as ``0111``.

Geographic Disaggregation
^^^^^^^^^^^^^^^^^^^^^^^^^

If location-specific CFs are provided, the method supports regionalized impact assessment.
Missing regions can fall back to broader aggregates:

- ``IT`` → ``RER`` → ``GLO``

Multiple CFs with overlapping geographic scopes can be combined
using their ``weight`` field.

Examples
--------

Minimal Example
^^^^^^^^^^^^^^^

.. code-block:: json

    {
      "name": "My LCIA Method",
      "version": "1.0",
      "unit": "kg CO2e",
      "exchanges": [
        {
          "supplier": {
            "name": "Carbon dioxide",
            "matrix": "biosphere"
          },
          "consumer": {
            "matrix": "technosphere"
          },
          "value": 1.0
        }
      ]
    }

Region-Specific Example
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

    {
      "supplier": {
        "name": "Carbon dioxide",
        "matrix": "biosphere"
      },
      "consumer": {
        "location": "FR",
        "matrix": "technosphere"
      },
      "value": 2.0
    }

Dynamic Expression Example
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

    {
      "supplier": {
        "name": "Methane, fossil",
        "operator": "contains",
        "matrix": "biosphere"
      },
      "consumer": {
        "matrix": "technosphere"
      },
      "value": "28 * (1 + 0.001 * (co2ppm - 410))"
    }

Country Pair Example
^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

    {
      "supplier": {
        "name": "aluminium production, primary",
        "reference product": "aluminium, primary",
        "location": "AL",
        "matrix": "technosphere",
        "operator": "startswith",
        "excludes": ["alloy", "market"]
      },
      "consumer": {
        "location": "GB",
        "matrix": "technosphere"
      },
      "value": 5.66e-08,
      "weight": 0.0038
    }

Saving and Loading
------------------

Save your custom LCIA method as a JSON file (e.g., ``my_method.json``),
then load it into ``edges`` using:

.. code-block:: python

    from edges import EdgeLCIA
    lcia = EdgeLCIA.from_file("my_method.json")

Make sure all required fields (`name`, `unit`, `version`, `exchanges`) are
present and correctly formatted.

---

This page can grow with more examples, including tips for debugging
mismatches or visualizing the CF structure. Let me know if you'd like
to include YAML support or template generators.
