function time_period_clustering(pf::PeriodsFinder)
    # Notation is based on the following paper:
    # [1] S. Pineda and J. M. Morales, “Chronological time-period clustering for optimal capacity expansion planning with storage,” IEEE Trans. Power Syst., vol. 33, no. 6, pp. 7162–7170, 2018.

    opts = pf.config["method"]
    type = recursive_get(opts, "clustering", "type", "hierarchical")
    N = get_number_of_periods(pf) # Initial number of clusters / periods
    NC = get_number_of_periods(pf) # Current number of clusters
    NCD = get_number_of_representative_periods(pf) # Desired number of cluster
    periods = get_set_of_periods(pf)
    norm_val = get_normalised_time_series_values(pf)
    S = get_set_of_time_series_names(pf)
    time_steps = get_set_of_time_steps(pf)
    ts_weights = get_time_series_weights(pf)
    err_weights = get_error_term_weights(pf)
    mandatory_periods = get_set_of_mandatory_periods(pf)
    ord_errs = get_set_of_ordering_errors(pf)
    ord_err_funcs = pf.inputs[:ordering_error_functions]

    ICM = [i for i in periods] # Set of periods which are the mediod of a cluster -> indices are clusters, elements are periods which are mediods of that cluster
    IP2C = [i for i in periods] # A set which maps the periods to their clusters -> indices are periods, elements are clusters
    IC2P = [[i] for i in periods] # A set which maps clusters to their periods -> indices are clusters, periods are elements
    x = [
        [Float64(ts_weights[s] * norm_val[s][i,t]) for s in S, t in time_steps][:]
        for i in periods
    ] # Cluster features vector
    xbar = copy(x) # Cluster centroids (i.e. averages)

    # TODO: Possible performance boosts:
    # 1. Changing the order of the for loops to be j then i ->
    # be careful with this though.
    # 2. Using @inbounds -> avoids check in the array bounds when running
    # 5. Can also try out @simd apparently?
    # 6. Get rid of if statements in loops - just make sure i != j

    # Define dissimilarity matrix
    # Since we only care about the (unordered) pairs,
    # we only define the upper / lower triangle of the matrix
    # For hierarchical clustering, D is lower triangular and we compute 1/D
    # For CTPC, D is upper triangular with only diagonals defined
    D = fill(Inf64, NC, NC)
    # NOTE: Can also define D as UpperTriangular
    # but this leads to 0 entries which can be annoying when
    # trying to find the minimum
    @fastmath for j in 1:NC
        if type == "chronological"
            istart = j
            iend = min(NC, j + 1)
        elseif type == "hierarchical"
            istart = j
            iend = NC
        end
        @fastmath for i in [i for i in istart:iend if i != j]
            num = Float64(2 * length(IC2P[i]) * length(IC2P[j]))
            den = Float64(length(IC2P[i]) + length(IC2P[j]))
            @inbounds D[i,j] = num / den * sum(
                ord_err_funcs[ord_err](xbar[i][:], xbar[j][:]) * err_weights[ord_err]
                for ord_err in ord_errs
            )
        end
    end
    if type == "hierarchical"
        # Do this so I can use findmax(D) and return lowest dissimilarity
        D = LowerTriangular(1 ./ D)
    end

    # Mandatory periods enforcement
    replaceVal = if type == "hierarchical"
        Float64(0.0)
    elseif type == "chronological"
        Inf64
    end
    for i in mandatory_periods
        D[i,:] .= replaceVal
    end
    for j in mandatory_periods
        D[:,j] .= replaceVal
    end

    # Correct for mandatory periods
    NCD = NCD - length(mandatory_periods)
    @assert iszero(NCD) == false "Can't run clustering if all periods are fixed."
    # TODO: could eventually replace this with a reasonable default.
    # TODO: currently mandatory periods don't represent other periods! Just themselves. I would say this is a limitation of clustering however
    ermsg = "Please allow for at least 1 period to be determined by the clustering algorithm."
    @assert NCD > 0 eval(ermsg)
    numAdj = get_number_of_clusters_of_adjacent_values(mandatory_periods)
    if type == "chronological"
        ermsg =
            """
            Too many periods fixed for this method. Please remove at least
            $(NCD - numAdj + 2) periods (I think...) 
            """
        @assert NCD > numAdj + 2 eval(ermsg)
    end

    # Get intermediate periods
    intermediate_periods = get_set_of_intermediate_periods(pf)

    # Create a progress meter
    NC0 = NC
    p = Progress(NC0 - NCD, 1.0)

    # Run clustering
    while NC - length(mandatory_periods) > NCD
        # Update progress meter
        update!(p, NC0 - NCD - NC - length(mandatory_periods))

        # Find the two "closest" mediods
        # This takes up half of the calculation time!
        if type == "chronological"
            # Only need to search along a diagonal of D!
            D_vec = [
                @inbounds D[j + 1,j] for j in 1:NC - 1 # ≈ 0.1 secs
            ]
            (val, idx) = findmin(D_vec) # ≈ 10^-5 secs

            # Convert this into an index for D (instead of D_vec)
            idx = CartesianIndex(idx + 1, idx) # ≈ 0.1 secs
        else
            (val, idx) = findmax(D) # ≈ 0.5 secs
        end

        # Merge the two clusters. Do this by appending the periods of the
        # "second" cluster to that of the "first" cluster
        IC2P[idx[1]] = append!(IC2P[idx[1]], IC2P[idx[2]])

        # Now we need to define the cluster mediod
        # To do this, need the centroid of the cluster and then need to
        # calculate dissimilarity of each period w.r.t to the cluster centroid
        # The period in the cluster which minimises this dissimalarity
        # becomes the next cluster mediod

        # Recalculate the cluster centroid
        sumterm = sum(hcat(x[IC2P[idx[1]][:]]...), dims=2)[:]
        xbar[idx[1]] = sumterm / length(IC2P[idx[1]])

        # Find the merged clusters mediod as follows:
        # Calculate the dissimilarity for each element in the cluster
        DC = [sum((xbar[idx[1]] .- el).^2) for el in x[IC2P[idx[1]][:]]]
        DC = [
            sum(
                ord_err_funcs[ord_err](xbar[idx[1]], el) * err_weights[ord_err]
                for ord_err in ord_errs
            )
            for el in x[IC2P[idx[1]][:]]
        ]

        # Find the period which minimises dissimilarity
        (val, idxClust) = findmin(DC)

        # Set this as the cluster mediod
        ICM[idx[1]] = periods[IC2P[idx[1]][idxClust]]

        # Get rid of the "second" cluster in the cluster
        deleteat!(IC2P, idx[2])
        deleteat!(xbar, idx[2])
        deleteat!(ICM, idx[2])

        # Update the mapping of periods to clusters
        for i = 1:length(IC2P)
            IP2C[IC2P[i][:]] .= i
        end

        # Now we update the dissimilarity matrix for the next iteration
        # Remove second cluster row and column from the dissimilarity matrix
        D_idx = [i for i in 1:NC if i != idx[2]]
        D = D[D_idx, D_idx]

        # Decrease number of clusters
        NC += -1

        # Change idx[1] in case it was the last cluster
        NC < idx[1] ? idx = CartesianIndex(NC, idx[2]) : nothing

        # Recompute the dissimilarity for the new cluster w.r.t to the other
        # clusters
        if type == "chronological"
            i_index = [
                i for i in max(1, idx[1] - 1):min(NC, idx[1])
            ]
            j_index = max.(1, i_index .- 1)
        elseif type == "hierarchical"
            i_index = vcat(
                [idx[1] for i = 1:idx[1] - 1],
                [i for i = idx[1]:NC]
            )
            j_index = vcat(
                [j for j = 1:idx[1] - 1],
                [idx[1] for j = idx[1]:NC]
            )
        end
        @assert length(i_index) == length(j_index)
        D_idx = [
            (i_index[k], j_index[k]) for k = 1:length(i_index)
            if isempty(intersect(IC2P[i_index[k]], mandatory_periods))
            && isempty(intersect(IC2P[j_index[k]], mandatory_periods))
            && i_index[k] != j_index[k]
        ]
        @fastmath for (i,j) in D_idx
            num = 2 * length(IC2P[i]) * length(IC2P[j])
            den = length(IC2P[i]) + length(IC2P[j])
            if type == "hierarchical"
                @inbounds D[i,j] = 1 ./ (num / den * sum(
                    ord_err_funcs[ord_err](xbar[i][:], xbar[j][:]) * err_weights[ord_err]
                    for ord_err in ord_errs
                    )
                )
            else
                @inbounds D[i,j] = num / den * sum(
                    ord_err_funcs[ord_err](xbar[i][:], xbar[j][:]) * err_weights[ord_err]
                    for ord_err in ord_errs
                )
            end
        end

        if NC in intermediate_periods
            dir = joinpath(get_abspath_to_result_dir(pf), string(NC))
            @info "Saving intermediate number of representative periods $NC to $dir"
            calculate_rep_periods_from_clusters!(pf, ICM, IC2P, IP2C)
            save(pf, dir)
        end
    end

    # Save the results
    calculate_rep_periods_from_clusters!(pf, ICM, IC2P, IP2C)
    return ICM, IC2P, IP2C, D
end

function calculate_rep_periods_from_clusters!(
        pf::PeriodsFinder, ICM, IC2P, IP2C
    )
    periods = get_set_of_periods(pf)
    pf.u = [i in ICM ? 1 : 0 for i in periods]
    pf.w = [i in ICM ? length(IC2P[IP2C[i]]) : 0 for i in periods]
    pf.v = zeros(length(periods), length(periods))
    for i in periods
        j = # I'm pretty sure this line is unnecessary
        pf.v[i,ICM[IP2C[i]]] = 1
    end
    rep_periods = sort(ICM)
    pf.v = pf.v[:, rep_periods]
    return pf
end