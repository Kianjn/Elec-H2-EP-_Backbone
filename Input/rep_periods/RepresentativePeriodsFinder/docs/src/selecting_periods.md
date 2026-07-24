# Selecting representative periods

## Overview

To select periods, the general workflow is:
```julia
using RepresentativePeriodsFinder, JuMP, Cbc
config_file = "/home/user/Desktop/selecting_periods/config_file.yaml"
pf = PeriodsFinder(config_file)
optimizer = optimizer_with_attributes(Cbc.Optimizer, "seconds" => 300) # seconds specifies time out limit
pf = find_representative_periods(pf, optimizer=optimizer)
```

Alternatively, you can just run:
```julia
config_file = "/home/user/Desktop/selecting_periods/config_file.yaml"
optimizer = optimizer_with_attributes(Cbc.Optimizer, "seconds" => 300)
pf = find_representative_periods(config_file, optimizer=optimizer)
```

The relevant options in the `.yaml` file may look something like this (assuming an optimisation based method):

```yaml
method:
  options: # Method agnostic options
    mandatory_periods: [] # Leave empty to specify none
    total_periods: 365
    representative_periods: 8
    time_steps_per_period: 24
    sampling_time: "Hour(1)" # This is a DateTime format - for quarter hours, Hour(0.25)
  
  optimization:
    integral_weights: false
    binary_ordering: true
    equal_weights: false

    duration_curve_error:
      weight: 0.5
      number_bins: 40
      type: "absolute" # Or "squared"
```

You do not need to specify an optimizer if you're using a clustering based method.

Running [`find_representative_periods(pf)`](@ref) will produce three outputs:
* `pf.u` (``u``) - a boolean vector indicating which periods have been selected to be representative.
* `pf.w` (``w``) - a vector of the weights attached to periods (0 if thse are not representative).
* `pf.v` (``v``) - a matrix mapping representative periods (columns) to non-representative periods (rows). Depending on the selection method chosen, this may not be defined.

## Options description

Default is indicated by an `=` sign.

!!! warning "Only specify `optimization` or `clustering` entries"
    Specifying both the `optimization` and `clustering` entries will lead to an error, as the absence of the entry is how [`find_representative_periods`](@ref) knows which method to use.

### Common options

* `mandatory_periods=[]`: Periods which will be forced to be selected. These are given a weight of 1 if clustering is used while the weight is a decision variable if optimisation is used.
* `total_periods`: Total number of periods ``N_{total}`` from which to select representative periods.
* `representative_periods`: Number of representative periods ``N_{rep}`` to be selected (or if these have already been selected, to be mapped to the rest of the periods throughout the year).
* `time_steps_per_period`: Number of time steps per period ``n_t = |T|``.
* `sampling_time`: Specify a different sampling time to that of the time series. Not recommended.
* `ordering_error`: Specify a vector of dictionaries with ordering errors / dissimilarity functions. These errors may be used in either the optimisation or clustering based methods and they may look something like the below. Here `x` and `y` are two vectors which are the time series values for two different periods (feature vectors in clustering speak). 
  * `weight=0`: The weight that an ordering error gets when selecting representative periods.

```yaml    
ordering_error: # aka dissimilarity metric for hierarchical clustering
    ord_err_1:
        function: "(x,y) -> sum(abs.( (sort(x) .- sort(y) )))"
        weight: 0.0
    ord_err_2:
        function: "(x,y) -> sum((x .- y).^2)"
        weight: 1.0
```

!!! warning "Age of the world errors"
    When specifying the ordering error functions as a string, Julia parses these and turns them into a function. This function creation has to happen before the call `find_representative_periods`, else you get an age of the world error. To avoid this, it is perhaps easier to specify the ordering error directly, as in the [Ramping and correlation time series](@ref) example.

### Optimisation methods

There is one general optimisation problem which is solved with differences coming from the objective terms and whether periods are selected or also ordered throughout the time series. Below is an example of the relevant options:

```yaml
optimization:
    integral_weights: false
    binary_ordering: true
    equal_weights: false

    time_series_error:
        weight: 0.5
        type: "squared" # Or "absolute"
    duration_curve_error:
        weight: 0.5
        number_bins: 40
        type: "squared" # Or "absolute"
```

!!! warning "Optimiser requirements"
    All methods require an optimiser capable of solving Mixed Integer Linear Programs (MILP), e.g. [`HiGHS`](https://github.com/jump-dev/HiGHS.jl), with some also requiring the ability to solve Mixed Integer Quadratic Programs (MIQP). If you only have access to the latter, be sure to always use `type: "absolute"` for the `time_series_error` and `duration_curve_error`. Also please consider bloody reading your solver error messages cogitating for a bit before whining to me.

* `integral_weights=false`: If `true`, the weights ``w`` (`pf.w`) given to the representative periods should be integers and not real numbers.
* `equal_weights=false`: If `true`, all weights of the representative periods must be equal to each other, i.e. equal to ``N_{total} / N_{rep}``.
* `binary_ordering=false`: If `true`, the ordering variable ``v`` (`pf.v`) consists of binaries instead of real numbers.

!!! warning "Binary ordering with ordering errors"
    If only ordering errors are specified, `binary_ordering` should be set to true, as the ordering error may be non-sensical otherwise. 
    
    For example, consider 3 vectors, ``x = [1,2]``, ``y=[3,4]`` and ``z=[-1,1]``. The absolute difference between these is ``D_{xy} = [2,2]`` , ``D_{xz} = [2,1]`` and ``D_{yz} = [4,3]``. If we create a new vector ``xy = 0.5x + 0.5y = [2,3]`` and take the absolute difference with ``z`` to give ``D_{xyz} = [3,2]`` then this is the same as ``0.5D_{xz} + 0.5D_{yz} = [3,2]``. Absolute value errors appear to work then (though don't quote me on that).

    Let's consider the same example, but with a squared difference (2-norm). ``D_{xy} = [4,4]``, ``D_{xz} = [4,1]`` and ``D_{yz} = [16,9]``. ``D_{xyz} = [9,4]`` then, which is not equal to ``0.5D_{xz} + 0.5D_{yz} = [10,5]``. Squared differences in this case do not seem to work.
    
    I (Sebastian Gonzato) am not sure exactly what properties an ordering error / dissimilarity function must have in order for linear combinations of ordering errors to be valid, but if you do then please let me know.

* `time_series_error`: If this entry is present, the time series error, i.e. the error between the time series values incurred by representing one period by another, is used to select representative periods. Generally discouraged for selecting representative periods due to computational complexity, though it is useful for ordering periods throughout a year once these have been selected (see [Re-ordering days in a year](@ref)).
  * `weight=0`: Weight assigned to this error in the objective function.
  * `type`: `"absolute"` for the absolute error between two time series or `"squared"` for the squared error (2-norm). The latter requires a solver which can solve MIQPs.
* `duration_curve_error`: If this entry is present, the duration curve error, i.e. the error incurred in approximating the original time series by selecting a period as representative and weighting it, is used to select representative periods. This is the "classic" method devised by [Poncelet et al.](https://www.mech.kuleuven.be/en/tme/research/energy_environment/Pdf/wp-2015-10b.pdf).
  * `weight`: see above.
  * `type`: see above, except error is between aggregated and discretised duration curve.
  * `number_bins`: Number of bins into which the duration curve is discretised.

### Clustering

You must specify your own dissimilarity function(s) in `options: ordering_error`.

```yaml
clustering: # Only one method (opt or clust) can be specified!
    type: "hierarchical"
    intermediate_periods: [] # specify intermediary results to save
```

* `type="hierarchical"`: If `"hierarchical"`, Ward's hierarchical clustering algorithm is used. If `"chronological"` then the same algorithm is used but only adjacent clusters can be merged so as to preserve chronology.
* `intermediate_periods=[]`: Specify intermediate numbers of clusters to save. For example, if you want to select 8, 16, 32 and 64 representative days, you could set `representative_periods=8` and `intermediate_periods=[16,32,64]`.
