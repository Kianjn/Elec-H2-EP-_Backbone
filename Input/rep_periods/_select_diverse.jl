# Pick the N most-different weather years out of the 10 cached ERA5 years.
# Uses ONE common NL calibration (derived from the reference year) for every
# year, so differences reflect genuine weather variability, not scaling.
using JSON, Statistics

here = @__DIR__
cache = joinpath(here, "weather_cache")

const REF_YEAR = 2015
const SOLAR_TARGET = 0.18
const WIND_TARGET  = 0.28

src = Dict(1=>2015,2=>2010,3=>2012,4=>2013,5=>2014,6=>2016,7=>2017,8=>2018,9=>2019,10=>2011)
labels = 1:10

function wind_speed_to_cf(speed_kmh)
    v_ci, v_r, v_co = 3.0, 12.0, 25.0
    cf = zeros(length(speed_kmh))
    for i in eachindex(speed_kmh)
        v = max(speed_kmh[i]/3.6, 0.0)
        if v >= v_ci && v < v_r
            cf[i] = ((v - v_ci)/(v_r - v_ci))^3
        elseif v >= v_r && v < v_co
            cf[i] = 1.0
        end
    end
    cf
end
radiation_to_cf(rad) = clamp.(rad ./ 1000.0, 0.0, 1.0)

function raw_year(y)
    d = JSON.parse(read(joinpath(cache, "open_meteo_$(y).json"), String))["hourly"]
    n = min(length(d["wind_speed_100m"]), 8760)
    (solar = radiation_to_cf(Float64.(d["shortwave_radiation"])[1:n]),
     wind  = wind_speed_to_cf(Float64.(d["wind_speed_100m"])[1:n]),
     temp  = Float64.(d["temperature_2m"])[1:n])
end

# Bisect a single multiplier so the REFERENCE year hits its NL annual CF target.
function calib_multiplier(cf, target)
    lo, hi = 0.0, 50.0
    for _ in 1:60
        mid = (lo+hi)/2
        mean(clamp.(cf .* mid, 0, 1)) < target ? (lo = mid) : (hi = mid)
    end
    hi
end

ref = raw_year(REF_YEAR)
m_solar = calib_multiplier(ref.solar, SOLAR_TARGET)
m_wind  = calib_multiplier(ref.wind,  WIND_TARGET)
println("Common NL calibration multipliers: solar=", round(m_solar,digits=4),
        "  wind=", round(m_wind,digits=4), "  (from ", REF_YEAR, ")\n")

years = Dict(l => raw_year(src[l]) for l in labels)
cal = Dict(l => (solar = clamp.(years[l].solar .* m_solar, 0, 1),
                 wind  = clamp.(years[l].wind  .* m_wind,  0, 1),
                 temp  = years[l].temp) for l in labels)

function features(l)
    s, w, t = cal[l].solar, cal[l].wind, cal[l].temp
    n = length(s); nd = n ÷ 24
    month_of = [min(12, ((i-1) ÷ 24) ÷ 30 + 1) for i in 1:n]
    fs = [mean(s[findall(==(m), month_of)]) for m in 1:12]
    fw = [mean(w[findall(==(m), month_of)]) for m in 1:12]
    dw = [mean(w[(d-1)*24+1:d*24]) for d in 1:nd]
    ds = [mean(s[(d-1)*24+1:d*24]) for d in 1:nd]
    dunkel = mean((dw .< 0.10) .& (ds .< 0.05))
    hdd = mean(max.(15.5 .- t, 0.0))      # heating degree-hours (demand driver)
    cdd = mean(max.(t .- 22.0, 0.0))      # cooling degree-hours
    vcat(fs, fw, [mean(s), mean(w), std(w), std(s), dunkel, hdd, cdd])
end

F = hcat([features(l) for l in labels]...)'
mu = mean(F, dims=1); sd = std(F, dims=1); sd[sd .< 1e-12] .= 1.0
Z = (F .- mu) ./ sd
D = [sqrt(sum((Z[i,:] .- Z[j,:]).^2)) for i in 1:10, j in 1:10]

# Exhaustive search: the 5-subset maximizing the MINIMUM pairwise distance.
function best_subset(D, k, n)
    best = (-Inf, Int[])
    for c in Iterators.product(ntuple(_->1:n, k)...)
        v = collect(c)
        issorted(v, lt=<) && length(unique(v)) == k || continue
        m = minimum([D[v[i], v[j]] for i in 1:k for j in i+1:k])
        m > best[1] && (best = (m, v))
    end
    return best
end
best = best_subset(D, 5, 10)
sel = best[2]

const IDX = Dict(:solar=>25, :wind=>26, :dunkel=>29, :hdd=>30, :cdd=>31)
println("Most-different 5 (max-min pairwise distance = ", round(best[1],digits=3), "): ", sel)
println("  ERA5 source years: ", [src[s] for s in sel], "\n")

println("label  src   solarCF  windCF  dunkel%  HDD-idx  CDD-idx   selected")
for l in labels
    println(rpad(l,6), rpad(src[l],6),
        rpad(round(F[l,IDX[:solar]],digits=4),9),
        rpad(round(F[l,IDX[:wind]],digits=4),8),
        rpad(round(100*F[l,IDX[:dunkel]],digits=2),9),
        rpad(round(F[l,IDX[:hdd]],digits=3),9),
        rpad(round(F[l,IDX[:cdd]],digits=4),10),
        l in sel ? "  <== YES" : "")
end
