![RepresentativePeriodsFinder logo](docs/src/assets/logo.png "RepresentativePeriodsFinder logo")

# RepresentativePeriodsFinder

This is a Julia package to select representative periods from time series data, typically to then be used in capacity expansion planning models. For more information see the [documentation](https://ucm.pages.gitlab.kuleuven.be/representativeperiodsfinder.jl/).

## Installation

* Download and install Julia from [here](https://julialang.org/downloads/).
* Assuming you've added Julia to your path, you can open up a Julia REPL by typing `julia` in a terminal.
* `RepresentativePeriodsFinder` can be added via the Julia package manager (type `]`): `pkg> add https://gitlab.kuleuven.be/UCM/representativedaysfinder.jl`

## Installation using docker
* make sure Docker is running
  ```bash
    docker build -t representativeperiodsfinder.jl:latest .
    docker run -it representativeperiodsfinder.jl:latest
  ```

## Basic usage

```julia
using Pkg
]add "https://gitlab.kuleuven.be/UCM/representativedaysfinder.jl"
using RepresentativePeriodsFinder, Cbc
config_file = <path_to_config_file>
find_representative_periods(config_file, optimizer=Cbc.Optimizer)
```
For more information see the [documentation](https://ucm.pages.gitlab.kuleuven.be/representativeperiodsfinder.jl/).
## Troubleshooting

See [troubleshooting](https://ucm.pages.gitlab.kuleuven.be/representativeperiodsfinder.jl/troubleshooting/).

## Reporting issues and feature requests

If you have any issues or feature requests, report them on GitLab and mention `@steffen.kaminski` or `@u0128861`.

## Alternative packages

* [TimeSeriesClustering.jl](https://holgerteichgraeber.github.io/TimeSeriesClustering.jl/stable/quickstart/)

## Developers

```julia
ENV["JULIA_PKG_DEVDIR"] = "<path_to_dir_of_choice>"
]dev "https://gitlab.kuleuven.be/UCM/representativedaysfinder.jl"
```

## Acknowledgements

This package is largely based on the [paper](https://www.mech.kuleuven.be/en/tme/research/energy_environment/Pdf/wp-2015-10b.pdf) and work of Kris Poncelet and Hanspeter Höschle where optimisation instead of clustering techniques were used to select representative days. This method was initially implemented in GAMS. An first Julia version was later implemented by Hanspeter Höschle while he was working at [VITO](https://vito.be/en). This was further developed by Sebastian Gonzato for a [paper on long term storage](https://www.mech.kuleuven.be/en/tme/research/energy-systems-integration-modeling/pdf-publications/wp-esim2021-1).

## License

This project is licensed under the GNU General Public License (GPL) v3.0 or later - see the `LICENSE.md` file for details.
