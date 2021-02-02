# plumes [![CircleCI](https://circleci.com/gh/icepack/plumes.svg?style=svg)](https://circleci.com/gh/icepack/plumes) [![codecov](https://codecov.io/gh/icepack/plumes/branch/master/graph/badge.svg)](https://codecov.io/gh/icepack/plumes)

This package contains solvers for various models of buoyant meltwater plumes under floating ice shelves.
The solvers are implemented using the finite element modeling package [Firedrake](https://www.firedrakeproject.org).

### Code structure

The library is structured roughly as follows:

    plumes
    ├── coefficients.py         # Physical constants
    ├── numerics.py             # Time discretization schemes
    └── models/
        ├── forms.py            # Spatial discretization helper functions
        ├── advection.py        # Scalar advection equation; for testing
        ├── shallow_water.py    # Nonlinear shallow water equations
        └── plume.py            # Shallow water + salt and heat transport

The directory `demo/` contains several Jupyter notebooks that show the features of this package.

### References

##### Physics

* Jenkins (1991). A One-Dimensional Model of Ice Shelf-Ocean Interaction. JGR
* Jenkins et al. (2010). Observation and Parameterization of Ablation at the Base of Ronne Ice Shelf, Antarctica. JPO
* Lazeroms et al. (2018). Modelling present-day basal melt rates for Antarctic ice shelves using a parametrization of buoyant meltwater plumes. The Cryosphere
* Favier et al. (2019). Assessment of sub-shelf melting parameterisations using theocean–ice-sheet coupled model NEMO(v3.6)–Elmer/Ice(v8.3). GMD
* Hoffman et al. (2019). Effect of Subshelf Melt Variability on Sea Level Rise Contribution From Thwaites Glacier, Antarctica. JGR Earth Surface

##### Observations

* Chartrand and Howat (2020). Basal channel evolution on the Getz Ice Shelf, West Antarctica. JGR Earth Surface
* Das et al. (2020). Multidecadal Basal Melt Rates and Structure of the Ross Ice Shelf. JGR Earth Surface. (see also [the data](https://www.usap-dc.org/view/dataset/601242) on USAP website)

##### Numerics

* Aizinger, Dawson (2002). A discontinuous Galerkin method for two-dimensional flow and transport in shallow water. Advances in Water Resources
* Dolejší, Feistauer (2015). *Discontinuous Galerkin Method: Analysis and Applications to Compressible Flow*. Springer
* Wimmer et al. (2020). Energy conserving upwinded compatible finite element schemes for the rotating shallow water equations. JCP
