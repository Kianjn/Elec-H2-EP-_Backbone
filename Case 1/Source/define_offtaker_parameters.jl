# ==============================================================================
# DEFINE OFFTAKER AGENT PARAMETERS (AMMONIA PRODUCERS)
# ==============================================================================
# This function defines parameters specific to Offtaker agents (Ammonia Producers).
# It handles two types of agents:
#   1. Electrolytic Ammonia Producer (Green): Buys H2 + H2 GCs, produces green ammonia
#   2. Grey Ammonia Producer: Produces ammonia via SMR, must buy H2 GCs for mandate
#
# The function processes configuration data to create agent-specific parameters
# such as capacity limits, conversion factors, and cost parameters.
#
# Arguments:
#   - agent_id::String: Unique identifier for the agent (e.g., "Offtaker_Green")
#   - model::JuMP.Model: JuMP optimization model container for this agent
#   - data::Dict: Agent-specific configuration dictionary from data.yaml
#                 Contains: Type, Alpha, Capacity_H2_In, ProcessingCost, etc.
#   - ts::Dict: Dictionary of time series DataFrames, keyed by year (not used here, but kept for consistency)
#   - repr_days::Dict: Dictionary of representative day DataFrames (not used here, but kept for consistency)
#
# Returns:
#   - Modifies the JuMP model in-place by storing agent-specific parameters
# ==============================================================================
function define_offtaker_parameters!(agent_id::String, model::JuMP.Model, data::Dict, ts::Dict, repr_days::Dict)

    # --- 1. ELECTROLYTIC AMMONIA PRODUCER LOGIC (GREEN OFFTAKER) ---
    # Green offtakers use green hydrogen to produce green ammonia
    # Process: Buy H2 + H2 GCs -> Convert to Ammonia -> Sell EP
    # Check if the agent type is "GreenOfftaker"
    if data["Type"] == "GreenOfftaker"
        # Store the conversion factor alpha (Output = alpha * Input)
        # This represents the stoichiometric efficiency: yield of NH3 per unit of H2
        # Example: If alpha = 1.0, then 1 MWh H2 produces 1 MWh NH3
        # Typical values: 0.8-1.0 depending on process efficiency
        # Units: MWh_NH3 / MWh_H2
        model[:alpha] = data["Alpha"]
        
        # Store the Input Capacity Limit (Maximum Hydrogen Intake in MW)
        # This represents the maximum hydrogen that can be received/processed
        # Limited by pipeline capacity, storage, or processing infrastructure
        # Uses 'get' with a default of infinity if not specified (though usually required)
        # Units: MW (hydrogen input)
        model[:H_buy_bar] = get(data, "Capacity_H2_In", Inf)
        
        # Store the Output Capacity Limit (Maximum End Product Production in MW)
        # This represents the maximum ammonia production capacity of the facility
        # Limited by factory size, equipment capacity, etc.
        # Units: MW (End Product output)
        model[:EP_sell_bar] = get(data, "Capacity_EP_Out", Inf)
        
        # Store the processing cost (non-fuel variable cost)
        # This represents the cost of converting hydrogen to ammonia
        # Includes: variable O&M, catalysts, utilities (excluding hydrogen feedstock cost)
        # The hydrogen feedstock cost is handled separately via H2 market prices
        # Units: EUR/MWh_EP
        model[:C_proc] = data["ProcessingCost"]

    # --- 2. GREY AMMONIA PRODUCER LOGIC ---
    # Grey offtakers produce ammonia using conventional means (SMR from natural gas)
    # Process: Natural Gas -> SMR -> Ammonia -> Sell EP
    # Must buy H2 GCs to meet 42% policy mandate (based on internal H2 consumption)
    # Check if the agent type is "GreyOfftaker"
    elseif data["Type"] == "GreyOfftaker"
        # Store the Production Capacity Limit (MW_EP)
        # This represents the maximum ammonia production capacity
        # Note: We map the generic "Capacity" key from data.yaml to the specific EP_sell_bar variable
        # This is the nameplate capacity of the ammonia production facility
        # Units: MW (End Product output)
        model[:EP_sell_bar] = data["Capacity"]
        
        # Store the marginal cost of production (EUR/MWh_EP) as the processing cost
        # This typically lumps together:
        #   - Natural gas input costs (feedstock for SMR)
        #   - Variable O&M costs
        #   - Other operational expenses
        # Note: This is the total marginal cost since grey producers don't buy H2 separately
        # The H2 is produced internally via SMR from natural gas
        # Units: EUR/MWh_EP
        model[:C_proc] = data["MarginalCost"]
        
        # Store the specific hydrogen intensity of grey ammonia production
        # This represents how much hydrogen (equivalent) is consumed internally per unit of ammonia
        # Grey ammonia production uses SMR to produce H2 from natural gas, then converts H2 to NH3
        # gamma_NH3: Specific hydrogen intensity (kg_H2/kg_NH3)
        # Typical value: ~0.18 kg H2 per kg NH3 for SMR-based production
        # This is used in the 42% mandate constraint: gc_h_buy_G >= 0.42 * (gamma_NH3 * ep_sell)
        # The mandate requires that 42% of the internal H2 consumption is green-certified
        # Units: kg_H2 / kg_NH3 (or equivalent in MWh units)
        # Default: 0.18 if not specified (typical value for SMR-based ammonia)
        model[:gamma_NH3] = get(data, "gamma_NH3", 0.18)

    # --- 3. END PRODUCT IMPORTER LOGIC (EP IMPORT) ---
    # This agent represents an external ammonia import option.
    # It supplies End Product to the EP market at a high but finite marginal cost,
    # without consuming hydrogen or H2 GCs. This adds flexibility and caps EP prices.
    # Check if the agent type is "EPImporter"
    elseif data["Type"] == "EPImporter"
        # Store the maximum import capacity (MW_EP)
        # This represents the maximum amount of ammonia that can be imported.
        # Units: MW (End Product output)
        model[:EP_sell_bar] = data["Capacity"]

        # Store the import cost as an effective processing cost
        # This is the marginal cost of imported ammonia (EUR/MWh_EP).
        # It should be higher than typical green/grey production costs to act as a price cap.
        model[:C_proc] = data["ImportCost"]

        # Flag this model as an importer so the build function can select the right structure.
        model[:is_importer] = true
    end
    
    # Print a confirmation message indicating parameters have been defined for this agent
    # This helps track the initialization progress during setup
    # Initialization print removed to reduce console noise
end
