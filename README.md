# ``edges``: Edge-based life cycle impact assessment

[![PyPI version](https://badge.fury.io/py/edges.svg)](https://badge.fury.io/py/csc-brightway)

``edges`` is a Python Library for exchange-based Impact Assessment in 
Life Cycle Analysis (LCA) for the ``brightway2``/``brightway25`` LCA framework.


The ``edges`` Python library offers a novel approach to applying characterization factors 
(CFs) during the impact assessment phase of Life Cycle Assessment (LCA). 
Unlike conventional methods that uniformly assign CFs to *nodes* (e.g., processes 
like ``Water, from well`` in the brightway2 ecosystem), ``edges`` shifts the focus to the 
*edges*—the *exchanges* or relationships between *nodes*. This allows CFs to be conditioned 
based on the specific context of each *exchange*. Essentially, ``edges`` introduces unique 
values in the characterization matrix tailored to the characteristics of each *edge*.

By focusing on *edges*, the library incorporates contextual information such as the 
attributes of both the *supplier* and the *consumer* (e.g., geographic location, ISIC 
classification, amount exchanged, etc.). This enables a more detailed and flexible 
impact characterization, accommodating parameters like the location of the consumer 
and the magnitude of the exchange.

Furthermore, ``edges`` supports the calculation of weighted CFs for both static regions 
(e.g., RER) and dynamic regions (e.g., RoW), enhancing its ability to model complex 
and region-specific scenarios.

## Features

- **National characterization factors** for water-related impacts.
- **Seamless integration** with the Brightway LCA framework.
- Implements national and sub-national characterization factors of:
  - the **AWARE method 1.2c**.
- Future updates will include additional impact categories.

## Installation

You can install the library using pip:

```bash
pip install git+https://github.com/Laboratory-for-Energy-Systems-Analysis/edges.git
```

## Getting Started

Check out the [examples' notebook](https://github.com/romainsacchi/edges/blob/main/examples/examples.ipynb).

### Check available methods from ``edges``

```python
    
from edges import get_available_methods

# Get the available methods
methods = get_available_methods()
print(methods)

```

### Perform edge-based LCIA with ``edges``

```python
import bw2data
from edges import EdgeLCIA

# Select an activity from the LCA database
act = bw2data.Database("ecoinvent-3.10-cutoff").random()

# Define a method
method = ('AWARE 1.2c', 'Country', 'unspecified', 'yearly')

# Initialize the LCA object
LCA = EdgeLCIA({act: 1}, method)
LCA.lci()
# Perform the LCAI calculation
LCA.lcia()
LCA.score

# Print a dataframe with the characterization factors used
LCA.generate_cf_table()

```

## Data Sources

* **AWARE**: The AWARE factors are adapted from peer-reviewed sources and tailored to provide 
precise country-specific data for environmental modeling. Refer to the AWARE 
website [https://wulca-waterlca.org/](https://wulca-waterlca.org/) for more information.

If you use the AWARE method, please cite the following publication:

```bibtex
@article{boulay2018aware,
  title={The WULCA consensus characterization model for water scarcity footprints: assessing impacts of water consumption based on available water remaining (AWARE).},
  author={Anne-Marie Boulay, Jane Bare, Lorenzo Benini, Markus Berger, Michael J. Lathuillière, Alessandro Manzardo, Manuele Margni, Masaharu Motoshita, Montserrat Núñez, Amandine Valerie Pastor, Bradley Ridoutt, Taikan Oki, Sebastien Worbe & Stephan Pfister },
  journal={Int J Life Cycle Assess},
  volume={23},
  pages={368–378},
  year={2018},
  publisher={Springer}
}
```

## Methodology

1. ``edges`` introduces edge-specific characterization factors
in the characterization matrix of ``bw2calc`` before performing the LCA calculation.
The characterization factors are stored in the ``data`` folder of the library. 
Currently, ``edges`` provides characterization factors for 346 national and 
sub-national regions, based on the [AWARE](https://wulca-waterlca.org/aware/) method,
based on the location of edge consumers. 

2. For specific ``ecoinvent`` regions (e.g., RER, Canada without Quebec, etc.), 
``edges`` computes the weighted average of the characterization factors for the 
countries included in the region, based either on population or GDP. The weighting 
key can be selected by the user (weighting by population size by default).

3. For relative regions (e.g., RoW, RoE, etc.), ``edges`` dynamically defines the 
locations included in the region based on the mathing activities in the LCA database. 
The weighted average of the characterization factor of the geographies 
containd in the region is then computed accordingly.

### How It Works

1. **Off-Diagonal Targeting**:
   - The library identifies specific exchanges between suppliers and consumers in the technosphere (A matrix).
   - Characterization factors are applied selectively based on these relationships, introducing off-diagonal terms in the characterization matrix (C matrix).
   - For example, an activity in Region A using resources from Region B can have a distinct characterization factor that reflects the inter-regional transfer's environmental consequences.

2. **Conditional Characterization**:
   - The characterization factors are conditioned on the **location** (or other attributes) of the activities.
   - For instance, water scarcity impacts might depend on both the supplier's and consumer's geographical context, assigning higher weights to regions with water stress.

3. **Matrix Adjustment**:
   - The enriched C matrix now accounts for interactions between different regions and flows. 
   - Off-diagonal elements (C_ij, where i ≠ j) capture interdependencies, such as the environmental cost of resource transport or upstream emissions.


## Contributing
Contributions are welcome! Please follow these steps to contribute:

1. **Fork** the repository.
2. **Create** a new branch for your feature or fix.
3. **Commit** your changes.
4. **Submit** a pull request.


## License
This project is licensed under the MIT License.
See the [LICENSE.md](LICENSE.md) file for more information.

## Contact
For any questions or inquiries, please contact the project maintainer 
at [romain.sacchi@psi.ch](mailto:romain.sacchi@psi.ch).

## Contributors

- [Romain Sacchi](https://github.com/romainsacchi)
- [Alvaro Hahn Menacho](https://github.com/alvarojhahn)

## Acknowledgments
The development of this library was entirely supported by the French agency for 
Energy [ADEME](https://www.ademe.fr/), via the financing of the [HySPI](https://www.isige.minesparis.psl.eu/actualite/le-projet-hyspi/) project.
The HySPI project aims to provide a methodological framework to analyze and 
quantify, in a systemic and prospective manner, the environmental impacts of the 
decarbonization strategy of H2 production used by the industry in France.
