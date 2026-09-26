# SafeBound

Official implementation of SafeBound: Safety-Aware Lower-Bound Learning for RUL Prediction.

## Overview

SafeBound reformulates Remaining Useful Life (RUL) prediction as a constraint-aware learning problem by jointly learning:

- a conservative lower bound,
- a non-negative slack variable for point prediction refinement,
- an instance-dependent dual variable,
- and a degradation-informed monotonic health representation.

The framework is evaluated on C-MAPSS, N-CMAPSS, and NASA Battery datasets.
