"""
    make_periods_finder_model!(pf::PeriodsFinder, optimizer)

Builds a JuMP model to select and/or order representative periods and assigns it to `pf.m`.
"""
function make_periods_finder_model!(
    pf::PeriodsFinder, optimizer::MathOptInterface.OptimizerWithAttributes
)
    m = pf.m = JuMP.Model(optimizer)
    m.ext = Dict(
        :variables => Dict{Symbol,Any}(),
        :constraints => Dict{Symbol,Any}(),
        :expressions => Dict{Symbol,Any}(),
        :objective => nothing, # Redefine later
    )
    opt = pf.config["method"]["optimization"] # optimization specifc options

    # Strict variables
    make_selection_variable!(pf, m)
    make_weighting_variable!(pf, m)
    if has_ordering_error(pf) == true
        make_ordering_variable!(pf, m)
    else
        m.ext[:variables][:v] = SVC(0.0)
    end

    # Dependent error variables
    if has_duration_curve_error(pf)
        make_duration_curve_error_variable!(pf, m)
        define_duration_curve_error_variable!(pf, m)
    else # else set to 0
        m.ext[:variables][:duration_curve_error] = SVC(0.0)
    end

    if has_time_series_error(pf)
        make_time_series_error_variable!(pf, m)
        define_time_series_error_variable!(pf, m)
    else # else set to 0
        m.ext[:variables][:time_series_error] = SVC(0.0)
    end

    # Fix variables
    fix_selection_variable!(pf, m)

    # Constraints
    limit_number_of_selected_periods!(pf, m)
    restrict_non_zero_weights_to_selected_periods!(pf, m)
    enforce_yearly_equivalent_duration!(pf, m)
    if has_ordering_error(pf)
        restrict_non_zero_orderings_to_selected_periods!(pf, m)
        restrict_linear_combination_of_representative_periods!(pf, m)
        relate_ordering_to_weights!(pf, m)
    end
    enforce_minimum_weight!(pf, m)

    # Objective
    formulate_objective!(pf, m)

    return m
end

function make_periods_finder_model!(pf::PeriodsFinder)
    return error("Please specify an optimizer to be used.")
end

function make_periods_finder_model!(pf::PeriodsFinder, x::Any)
    return error(
        """
  Second argument must be an optimizer with attributes. 
  Perhaps you only specified e.g. `Cbc.Optimizer`, and not `optimizer_with_attributes(Cbc.Optimizer)`?
  """,
    )
end

function optimize_periods_finder_model!(pf::PeriodsFinder, m::JuMP.Model=pf.m)
    optimize!(m)
    stat = termination_status(m)

    if stat in
       [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.TIME_LIMIT, MOI.ITERATION_LIMIT] &&
        has_values(m)
        # Save selection variable as is
        pf.u = round.(var_value((m.ext[:variables][:u])))

        # Save all entries of w
        rep_periods = get_set_of_representative_periods(pf)
        w = var_value(m.ext[:variables][:w])
        if length(w) == length(rep_periods)
            pf.w = zeros(get_number_of_periods(pf))
            pf.w[rep_periods] = w
        elseif length(w) == get_number_of_periods(pf)
            pf.w = w
        else
            error(
                "Length of w, $(length(w)) is unexpected. Perhaps you were re-ordering and forgot to set the parameter `mandatory_periods`?",
            )
        end

        # Only save non-zero entries v to save space
        v = var_value(m.ext[:variables][:v])
        if typeof(v) <: SingleValuedContainer # i.e. was not defined
            v = get_educated_guess_for_ordering_variable(pf)
        end
        if size(v, 2) != length(rep_periods)
            pf.v = v[:, rep_periods]
        elseif size(v, 2) == length(rep_periods)
            pf.v = v
        else
            error("Size of v, $(size(v)) is unexpected.")
        end
    else
        @warn "Termination criteria is $stat and `has_values(m) = $(has_values(m))`, solution not saved to PeriodsFinder."
    end

    # Return solution status
    return stat
end

function make_selection_variable!(pf::PeriodsFinder, m::JuMP.Model)
    rep_periods = get_set_of_representative_periods(pf)
    mandatory_periods = get_set_of_mandatory_periods(pf)
    if length(rep_periods) == length(mandatory_periods)
        # Selection fully specified, just return a Boolean vector
        N_total = get_number_of_periods(pf)
        u = zeros(Bool, N_total)
        u[mandatory_periods] .= true
        return m.ext[:variables][:u] = u
    else
        return m.ext[:variables][:u] = @variable(
            m, [j in rep_periods], binary = true, base_name = "u"
        )
    end
end

function make_weighting_variable!(pf::PeriodsFinder, m::JuMP.Model)
    rep_periods = get_set_of_representative_periods(pf)
    opt = pf.config["method"]["optimization"]
    integer_bool = get(opt, "integral_weights", false)
    return m.ext[:variables][:w] = @variable(
        m,
        [j in rep_periods],
        integer = integer_bool,
        base_name = "w",
        lower_bound = 0.0
    )
end

function make_ordering_variable!(pf::PeriodsFinder, m::JuMP.Model)
    periods = get_set_of_periods(pf)
    rep_periods = get_set_of_representative_periods(pf)
    opt = pf.config["method"]["optimization"]
    bin_bool = get(opt, "binary_ordering", false)
    return m.ext[:variables][:v] = @variable(
        m,
        [i in periods, j in rep_periods],
        binary = bin_bool,
        base_name = "v",
        lower_bound = 0.0
    )
end

function make_duration_curve_error_variable!(pf::PeriodsFinder, m::JuMP.Model)
    S = get_set_of_time_series_names(pf)
    bins = get_set_of_bins(pf)
    return m.ext[:variables][:duration_curve_error] = @variable(
        m, [s in S, b in bins], base_name = "dc_err", lower_bound = 0.0
    )
end

function make_time_series_error_variable!(pf::PeriodsFinder, m::JuMP.Model)
    S = get_set_of_time_series_names(pf)
    periods = get_set_of_periods(pf)
    time_steps = get_set_of_time_steps(pf)
    return m.ext[:variables][:time_series_error] = @variable(
        m,
        [s in S, i in periods, t in time_steps],
        base_name = "ts_err",
        lower_bound = 0.0
    )
end

function fix_selection_variable!(pf::PeriodsFinder, m::JuMP.Model)
    u = m.ext[:variables][:u]
    mandatory_periods = get_set_of_mandatory_periods(pf)
    if eltype(u) <: JuMP.VariableRef
        for p in mandatory_periods
            fix(u[p], true; force=true)
        end
    end
    return u
end

function set_selection_variable_initial_guess!(pf::PeriodsFinder, m::JuMP.Model)
    # TODO: fill in
    return nothing
end

function limit_number_of_selected_periods!(pf::PeriodsFinder, m::JuMP.Model)
    rep_periods = get_set_of_representative_periods(pf)
    N_rep = get_number_of_representative_periods(pf)
    u = m.ext[:variables][:u]
    return m.ext[:constraints][:limit_num_periods] = @constraint(
        m, sum(u[j] for j in rep_periods) <= N_rep
    )
end

function restrict_non_zero_weights_to_selected_periods!(
    pf::PeriodsFinder, m::JuMP.Model
)
    opt = pf.config["method"]["optimization"]
    equal_weights_bool = get(opt, "equal_weights", false)
    N_total = get_number_of_periods(pf)
    N_rep = get_number_of_representative_periods(pf)
    rep_periods = get_set_of_representative_periods(pf)
    u = m.ext[:variables][:u]
    w = m.ext[:variables][:w]
    if equal_weights_bool == true
        con = [
            @constraint(m, w[j] == u[j] * N_total / N_rep) for j in rep_periods
        ]
    else
        con = [@constraint(m, w[j] <= u[j] * N_total) for j in rep_periods]
    end
    return m.ext[:constraints][:restrict_non_zero_weights] = con
end

function enforce_yearly_equivalent_duration!(pf::PeriodsFinder, m::JuMP.Model)
    rep_periods = get_set_of_representative_periods(pf)
    N_total = get_number_of_periods(pf)
    w = m.ext[:variables][:w]
    return m.ext[:constraints][:equivalent_yearly_duration] = @constraint(
        m, sum(w[j] for j in rep_periods) == N_total
    )
end

function restrict_non_zero_orderings_to_selected_periods!(
    pf::PeriodsFinder, m::JuMP.Model
)
    periods = get_set_of_periods(pf)
    rep_periods = get_set_of_representative_periods(pf)
    u = m.ext[:variables][:u]
    v = m.ext[:variables][:v]
    con = [@constraint(m, v[i, j] <= u[j]) for i in periods, j in rep_periods]
    return m.ext[:constraints][:restrict_ordering] = con
end

function restrict_linear_combination_of_representative_periods!(
    pf::PeriodsFinder, m::JuMP.Model
)
    opt = pf.config["method"]["optimization"]
    linear_comb = get(
        opt, "linear_combination_representative_periods", "sum_to_one"
    )
    if linear_comb == "sum_to_one"
        periods = get_set_of_periods(pf)
        rep_periods = get_set_of_representative_periods(pf)
        v = m.ext[:variables][:v]
        con = [
            @constraint(m, sum(v[i, j] for j in rep_periods) == 1) for
            i in periods
        ]
        return m.ext[:constraints][:linear_combination_representative_periods] =
            con
    end
    return nothing
end

function relate_ordering_to_weights!(pf::PeriodsFinder, m::JuMP.Model)
    opt = pf.config["method"]["optimization"]
    periods = get_set_of_periods(pf)
    rep_periods = get_set_of_representative_periods(pf)
    @unpack w, v = m.ext[:variables]
    con = [
        @constraint(m, sum(v[i, j] for i in periods) == w[j]) for
        j in rep_periods
    ]
    return m.ext[:constraints][:relate_ordering_to_weights] = con
end

function enforce_minimum_weight!(pf::PeriodsFinder, m::JuMP.Model)
    opt = pf.config["method"]["optimization"]
    rep_periods = get_set_of_representative_periods(pf)
    min_weight = get(opt, "minimum_weight", 0.0)
    @unpack w, u = m.ext[:variables]
    return m.ext[:constraints][:minimum_weight] = [
        @constraint(m, w[j] >= u[j] * min_weight) for j in rep_periods
    ]
end

function define_time_series_error_variable!(pf::PeriodsFinder, m::JuMP.Model)
    ts_err_opt = pf.config["method"]["optimization"]["time_series_error"]
    type = get(ts_err_opt, "type", "squared")
    S = get_set_of_time_series_names(pf)
    periods = get_set_of_periods(pf)
    rep_periods = get_set_of_representative_periods(pf)
    time_steps = get_set_of_time_steps(pf)
    N_total = get_number_of_periods(pf)
    x = get_normalised_time_series_values(pf)

    @unpack time_series_error, v = m.ext[:variables]

    if type == "squared"
        time_series_error = @constraint(
            m,
            [s in S, i in periods, t in time_steps],
            time_series_error[s, i, t] ==
                +sum(v[i, j] * x[s][j, t] for j in rep_periods) - x[s][i, t]
        )
        @pack! m.ext[:constraints] = time_series_error

    elseif type == "absolute"
        reconstructed_time_series = @expression(
            m,
            [s in S, i in periods, t in time_steps],
            sum(v[i, j] * x[s][j, t] for j in rep_periods)
        )
        time_series_error_eq1 = @constraint(
            m,
            [s in S, i in periods, t in time_steps],
            time_series_error[s, i, t] >=
                +reconstructed_time_series[s, i, t] - x[s][i, t]
        )
        time_series_error_eq2 = @constraint(
            m,
            [s in S, i in periods, t in time_steps],
            time_series_error[s, i, t] >=
                -reconstructed_time_series[s, i, t] + x[s][i, t]
        )
        @pack! m.ext[:expressions] = reconstructed_time_series
        @pack! m.ext[:constraints] = time_series_error_eq1,
        time_series_error_eq2
    end
    return m
end

function define_duration_curve_error_variable!(pf::PeriodsFinder, m::JuMP.Model)
    dc_err_opt = pf.config["method"]["optimization"]["duration_curve_error"]
    type = get(dc_err_opt, "type", "squared")
    S = get_set_of_time_series_names(pf)
    bins = get_set_of_bins(pf)
    rep_periods = get_set_of_representative_periods(pf)
    A = get_duration_curve_parameter(pf)
    L = get_discretised_duration_curve(pf)
    N_total = get_number_of_periods(pf)

    @unpack duration_curve_error, w = m.ext[:variables]

    if type == "squared"
        duration_curve_error = @constraint(
            m,
            [s in S, b in bins],
            duration_curve_error[s, b] ==
                +sum(w[j] / N_total * A[s][j, b] for j in rep_periods) - L[s][b]
        )
    elseif type == "absolute"
        duration_curve_error_eq1 = @constraint(
            m,
            [s in S, b in bins],
            duration_curve_error[s, b] >=
                +L[s][b] - sum(w[j] / N_total * A[s][j, b] for j in rep_periods)
        )
        duration_curve_error_eq2 = @constraint(
            m,
            [s in S, b in bins],
            duration_curve_error[s, b] >=
                -L[s][b] + sum(w[j] / N_total * A[s][j, b] for j in rep_periods)
        )
        @pack! m.ext[:constraints] = duration_curve_error_eq1,
        duration_curve_error_eq2
    end
    return m
end

function get_opt_error_func(type::String, error_str::String)
    if type == "absolute"
        return identity # x -> x 
    elseif type == "squared"
        return squared # x -> x^2
    else
        error(
            """Do not recognise $(error_str) of type "$(type)". Must be either "squared" or "absolute".""",
        )
    end
end

function formulate_objective!(pf::PeriodsFinder, m::JuMP.Model)
    opt = pf.config["method"]["optimization"]
    dc_err_type = recursive_get(opt, "duration_curve_error", "type", "absolute")
    ts_err_type = recursive_get(opt, "time_series_error", "type", "absolute")

    ordering_errors = get_set_of_ordering_errors(pf)
    objective_term_weights = get_error_term_weights(pf)
    time_series_weights = get_time_series_weights(pf)
    periods = get_set_of_periods(pf)
    rep_periods = get_set_of_representative_periods(pf)
    time_steps = get_set_of_time_steps(pf)
    bins = get_set_of_bins(pf)
    S = get_set_of_time_series_names(pf)

    ordering_error_expressions = Dict{String,Any}(
        ord_err => ordering_error_expression!(pf, m, ord_err) for
        ord_err in ordering_errors
    )

    @unpack duration_curve_error, time_series_error = m.ext[:variables]

    dc_err_func = get_opt_error_func(dc_err_type, "duration curve error")
    ts_err_func = get_opt_error_func(ts_err_type, "time series error")

    obj = @objective(
        m,
        Min,
        sum(
            time_series_weights[s] * (
                +objective_term_weights["duration_curve_error"] *
                sum(dc_err_func(duration_curve_error[s, b]) for b in bins) +
                objective_term_weights["time_series_error"] * sum(
                    ts_err_func(time_series_error[s, i, t]) for i in periods,
                    t in time_steps
                ) +
                sum(
                    objective_term_weights[ord_err] * sum(
                        ordering_error_expressions[ord_err][s][i, j] for
                        i in periods, j in rep_periods
                    ) for ord_err in ordering_errors
                )
            ) for s in S
        )
    )

    return m.ext[:objective] = obj
end

function ordering_error_expression!(
    pf::PeriodsFinder, m::JuMP.Model, err_name::String
)
    errorfunc = pf.inputs[:ordering_error_functions][err_name]
    periods = get_set_of_periods(pf)
    rep_periods = get_set_of_representative_periods(pf)
    S = get_set_of_time_series_names(pf)
    x = get_normalised_time_series_values(pf)
    v = m.ext[:variables][:v]

    ord_err_expr = Dict(
        s => @expression(
            m,
            [i in periods, j in rep_periods],
            v[i, j] * errorfunc(x[s][i, :], x[s][j, :])
        ) for s in S
    )

    @assert all([all(length.(ord_err_expr[s].data) .== 1) for s in S]) "$err_name must return scalar value."

    return m.ext[:expressions][Symbol(err_name)] = ord_err_expr
end
