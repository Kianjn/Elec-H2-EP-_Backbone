RepresentativePeriodsFinder release notes
==========================================

Version 0.4.4 (July 29th 202)
---------------------------------

- Beefed up documentation.
- Added more options for writing out results.
- Added an example to illustrate how to include ramping and correlation time series.
- Added warning if no objective present for period selection.
- Fixed time series weights not being applied bug.
- Created example for interfacing with [`TimeSeriesClustering.jl`](https://holgerteichgraeber.github.io/TimeSeriesClustering.jl/stable/quickstart/).
- Fixed bin discretisation error bug (occurred when there were many zero values, e.g. for solar, see [#30](https://gitlab.kuleuven.be/UCM/representativeperiodsfinder.jl/-/issues/30)).
- Fixed time series error bug (ordering was not occurring properly which is probably why the days re-ordering example wasn't working properly).
- Fixed `mkrootdirs` being OS dependent.
- Fixed `Int64` type assertions for Julia versions >1.7.

Version 0.4.3 (August 9th 2021)
---------------------------------

- Fixed non unique time stamp bug.
- Fixed bin discretisation error bug.
- Fixed plotting offset issue.
- Minor edits of documentation and `README` file.
- Cleaned up dependencies and updated packages.

Version 0.4.2 (May 27th 2021)
---------------------------------

- Fixed ordering error bug
- Added example for re-ordering

Version 0.4.1 (March 24th 2021)
---------------------------------

- There was an error in the optimisation formulation when re-ordering, this was fixed.
- Fixed deployment of documentation.
- Restricted versions of DataFrames and CSV since these can cause issues. Also added entries about this in README.md.
- Extended tests to be performed also for Julia 1.2.0, 1.3.0 and 1.4.0 (but only on the master branch).
- Changed `Getting Started` in documentation to allow for copy pasting the `.yaml` configuration file.

Version 0.4.0 (February 15th 2021)
---------------------------------

- General overhaul of code and increased documentation.
