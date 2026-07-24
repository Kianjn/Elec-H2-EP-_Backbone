# RepresentativePeriodsFinder.jl

[`RepresentativePeriodsFinder.jl`](https://gitlab.kuleuven.be/UCM/representativeperiodsfinder.jl) is a Julia package to select representative periods from time series data, usually for use in capacity expansion planning models. While this is typically done using clustering techniques (see for example [TimeSeriesClustering.jl](https://holgerteichgraeber.github.io/TimeSeriesClustering.jl/stable/quickstart/)), this package is unique in that it can solve optimisation problems to select representative periods. For more information on this see these two papers:
* [Poncelet, K., Höschle, H., Delarue, E., Virag, A., D'haeseleer, W. 2016. Selecting representative days for capturing the implications of integrating intermittent renewables in generation expansion planning problems. Accepted for publication in IEEE Transactions on power systems.](https://www.mech.kuleuven.be/en/tme/research/energy_environment/Pdf/wp-2015-10b.pdf)
* [Gonzato, S., Bruninx, K., Delarue, E., 2021. Long term storage in generation expansion planning models with a reduced temporal scope. Working paper.](https://www.mech.kuleuven.be/en/tme/research/energy-systems-integration-modeling/pdf-publications/wp-esim2021-1)

## Installation

`RepresentativePeriodsFinder.jl` can be added via the Julia package manager (type `]`):
```julia
] add https://gitlab.kuleuven.be/UCM/representativedaysfinder.jl
```

## Getting Started
See the [Getting Started](@ref getting_started_page)

## Reporting issues and feature requests

If you have any issues or feature requests, report them on GitLab and mention `@steffen.kaminski` or `@u0128861`. Most of the GitLab issues already there are just waiting for someone to motivate their implementation.
