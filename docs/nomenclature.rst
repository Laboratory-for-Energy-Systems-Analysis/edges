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

Weight Field
^^^^^^^^^^^^

The optional ``weight`` field is used when multiple CFs are available for a supplier-consumer pair (e.g., in regionalized methods). Weights are normalized and used to compute a weighted average.

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
