# ==============================================================================
# UPDATE ADMM PENALTY PARAMETER (RHO) - ADAPTIVE STRATEGY
# ==============================================================================
# This function updates the ADMM penalty parameter (rho) adaptively based on the
# relative magnitudes of primal and dual residuals. This is based on the standard
# approach by Boyd et al. to balance primal and dual convergence.
#
# Adaptive Rho Strategy:
#   - If Primal Residual >> Dual Residual: Increase rho (stronger penalty for imbalances)
#   - If Dual Residual >> Primal Residual: Decrease rho (weaker penalty, allow price adjustments)
#   - Goal: Balance both types of convergence for optimal algorithm performance
#
# Why Adaptive Rho?
#   - Fixed rho may lead to slow convergence or oscillations
#   - Adaptive rho helps the algorithm converge faster by adjusting penalty strength
#   - Higher rho = faster constraint satisfaction (primal convergence)
#   - Lower rho = better price stability (dual convergence)
#
# Arguments:
#   - elec::Dict: Electricity market dictionary (modified in-place)
#   - h2::Dict: Hydrogen market dictionary (modified in-place)
#   - elec_gc::Dict: Electricity GC market dictionary (modified in-place)
#   - h2_gc::Dict: Hydrogen GC market dictionary (modified in-place)
#   - ep::Dict: End Product market dictionary (modified in-place)
#   - primal::Float64: Current primal residual (maximum absolute imbalance)
#   - dual::Float64: Current dual residual (stationarity measure)
#   - params::Dict: ADMM parameters dictionary (not used here, kept for interface consistency)
#
# Returns:
#   - Modifies all market rho values in-place (if adjustment is needed)
# ==============================================================================
function update_rho!(elec, h2, elec_gc, h2_gc, ep, primal, dual, params)
    
    # --- 1. DEFINE ADAPTIVE PARAMETERS ---
    # These parameters control when and how much to adjust rho
    
    # mu: Threshold ratio to trigger an update
    # If one residual is > mu times the other, we adjust rho
    # Example: If mu = 10.0 and primal > 10 * dual, we increase rho
    # Typical values: 5.0 - 20.0
    mu = 10.0
    
    # tau_incr: Factor to increase rho (make penalty stricter)
    # When primal >> dual, we multiply rho by tau_incr
    # Example: If tau_incr = 2.0, rho doubles
    # Typical values: 1.5 - 3.0
    tau_incr = 2.0
    
    # tau_decr: Factor to decrease rho (loosen penalty)
    # When dual >> primal, we divide rho by tau_decr
    # Example: If tau_decr = 2.0, rho halves
    # Typical values: 1.5 - 3.0
    tau_decr = 2.0
    
    # Initialize the scaling factor to 1.0 (no change)
    # This will be set to tau_incr or 1/tau_decr if adjustment is needed
    new_factor = 1.0

    # --- 2. CHECK CONDITIONS FOR RHO ADJUSTMENT ---
    
    # Condition A: Primal Residual is significantly larger than Dual Residual
    # Implication: Constraints (Supply=Demand) are being violated too much
    #              compared to convergence stationarity.
    #              The markets are far from equilibrium, but prices are relatively stable.
    # Action: Increase rho to enforce constraints more strictly.
    #         This will penalize imbalances more heavily, encouraging faster market clearing.
    if primal > mu * dual
        # Set scaling factor to increase rho
        new_factor = tau_incr
        
    # Condition B: Dual Residual is significantly larger than Primal Residual
    # Implication: Decision variables are changing too rapidly/oscillating
    #              compared to the constraint violation.
    #              Prices are oscillating even though markets are relatively balanced.
    # Action: Decrease rho to reduce the penalty weight and let dual variables
    #         (prices) guide the convergence more smoothly.
    #         This reduces oscillations and improves solution stability.
    elseif dual > mu * primal
        # Set scaling factor to decrease rho (divide by tau_decr)
        new_factor = 1.0 / tau_decr
    end

    # --- 3. APPLY RHO UPDATE TO MARKETS ---
    # Only perform multiplication if the factor changed (efficiency check)
    # If new_factor == 1.0, no adjustment is needed, so we skip the update
    # 
    # IMPORTANT: We only update base markets (Electricity, Hydrogen, EP) with adaptive rho.
    # GC markets (Electricity_GC, Hydrogen_GC) are kept at their fixed low rho values
    # to maintain the decoupling strategy. If we scale GC markets, we destroy the
    # carefully tuned rho ratios that break the coupling between base and GC markets.
    if new_factor != 1.0
        # Update Electricity Market rho (base market - can adapt)
        elec["rho"] *= new_factor
        
        # Update Hydrogen Market rho (base market - can adapt)
        h2["rho"] *= new_factor
        
        # Update End Product Coordination rho (base market - can adapt)
        ep["rho"] *= new_factor
        
        # NOTE: GC markets are NOT updated here to preserve their low rho values
        # This maintains the decoupling strategy where GC markets have lower rho
        # to allow base markets to stabilize first, then GC markets follow.
        # If we scale GC markets, we would destroy the carefully tuned ratios.
        # 
        # Electricity GC Market: Keep at fixed low rho (e.g., 0.3)
        # Hydrogen GC Market: Keep at fixed low rho (e.g., 0.1)
        # These values are set in data.yaml and should remain constant.
        
        # Optional: Print log for debugging adaptive steps
        # Uncomment the line below to see when and how rho is adjusted
        # This is useful for understanding algorithm behavior and tuning parameters
        # println("Updated Rho by factor $new_factor. New Rho_E: $(elec["rho"]), Rho_H2: $(h2["rho"])")
    end
end
