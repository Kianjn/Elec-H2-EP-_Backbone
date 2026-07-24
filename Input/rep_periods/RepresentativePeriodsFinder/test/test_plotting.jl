using RepresentativePeriodsFinder, JuMP, Cbc
RPF = RepresentativePeriodsFinder

# Create PeriodsFinder
config_file = RPF.datadir("default.yaml")
pf = PeriodsFinder(config_file, populate_entries=true)

# Set up a random selection of days
N_total = RPF.get_number_of_periods(pf)
pf.u = rand(Bool, N_total)
rep_periods = RPF.get_set_of_representative_periods(pf)
N_repr = length(rep_periods)
pf.config["method"]["options"]["representative_periods"] = N_repr
pf.v = zeros(Bool, N_total, N_total)
for j in rep_periods
    pf.v[j, j] = 1
end
for i in setdiff(1:N_total, rep_periods)
    rand_vec = zeros(size(pf.v, 2))
    rand_idx = rand(setdiff(1:N_total, rep_periods))
    rand_vec[rand_idx] = 1
    pf.v[i,:] = rand_vec
end
pf.w = sum(pf.v, dims=1)[:]
pf.v = pf.v[:,rep_periods]

# Plot this
RPF.create_plots(pf)