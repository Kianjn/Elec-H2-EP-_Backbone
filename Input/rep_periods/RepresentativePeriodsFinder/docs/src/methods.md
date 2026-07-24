# Methods

The optimisation and clustering methods which are implemented here are adapted from in [Long term storage in generation expansion planning models with a reduced temporal scope by Gonzato, S., Bruninx, K., Delarue, E., 2021.](https://www.mech.kuleuven.be/en/tme/research/energy-systems-integration-modeling/pdf-publications/wp-esim2021-1).

## Optimisation

The optimisation problem which is solved can be written as follows:

```math
\begin{align*}
    \min_{u_i, w_i, v_{ij}} & \sum_{s \in \mathcal{S}} W_s \cdot \sum_{e \in \mathcal{E}} W_e \cdot f_e(u_i, w_i, v_{ij}) \\
    s.t. & \\
    & u_i \leq N_{rep} \\
    & \sum_{i \in \mathcal{P}} w_i \leq N_{total} \\
    & \sum_{j \in \mathcal{RP}} w_i \leq N_{total} \\
    & \sum_{j \in \mathcal{RP}} v_{ij} = 1 \quad i \in \mathcal{P} \\
    & \sum_{i \in \mathcal{P}} v_{ij} = w_i \quad j \in \mathcal{RP} \\
    & v_{ij} \leq u_{i} \quad i \in \mathcal{P}, \; j \in \mathcal{RP} \\
    & w_i \leq u_{i} \cdot N_{total} \quad i \in \mathcal{P} \\
    & u_i \in {0,1}, \quad w_i \in \R^+_0, \quad v_{ij} \in [0,1]
\end{align*}
```

### Sets:

* ``\mathcal{S}`` - the set of time series.
* ``\mathcal{E}`` - the set of error functions.
* ``\mathcal{P}`` - the set of all periods.
* ``\mathcal{RP}`` - the set of representative periods (which is equal to the set of all periods unless they have already been selected).

### Variables:

* ``u_i`` - selection variable, equal to 1 if period ``i`` is selected as representative and 0 otherwise.
* ``w_i`` - weighting variable, greater than 0 if period ``i`` is selected as representative and 0 otherwise.
* ``v_{ij}`` - ordering variable, greater than 0 if period ``i`` is represented by period `j` and 0 otherwise.

### Parameters

* ``W_s`` - weighting of time series ``s``.
* ``W_e`` - weighting of error function ``e``.
* ``N_{total}`` - total number of periods.
* ``N_{rep}`` - number of representative periods to be selected.

#### Error functions

There are 3 principal types of error functions:

* Time series error functions, ``f(v_{ij})``.
* Duration curve error functions, ``f(w_{ij})``.
* Ordering error functions, ``f(v_{ij})``

The ordering error functions are user defined (see [Selecting representative periods](@ref)). For the time series and duration curve error functions, the reader is referred to [`optimization.jl` file](https://gitlab.kuleuven.be/UCM/representativeperiodsfinder.jl/-/blob/dev/src/methods/optimization.jl), as describing these is prohibitively time consuming. The time series error is briefly described in [Gonzato et al.](https://www.mech.kuleuven.be/en/tme/research/energy-systems-integration-modeling/pdf-publications/wp-esim2021-1) and the duration curve error in [Poncelet et al](https://www.mech.kuleuven.be/en/tme/research/energy_environment/Pdf/wp-2015-10b.pdf).

## Clustering

Ward's hierarchical clustering method is used. This method is well described [here](https://www.mech.kuleuven.be/en/tme/research/energy-systems-integration-modeling/pdf-publications/wp-esim2021-1) and [here](https://www.researchgate.net/publication/325459549_Chronological_Time-Period_Clustering_for_Optimal_Capacity_Expansion_Planning_With_Storage).

One particularity with the algorithm is the ability to include mandatory periods. This is done by fixing the dissimilarity for the mandatory period with respect to the all other periods to `Inf`. This ensures that it is not merged. It is not possible to assign this period any other weight than one however. 

Another particularity is the ability to choose the dissimilarity function (ordering error) freely (see [Selecting representative periods](@ref)).
