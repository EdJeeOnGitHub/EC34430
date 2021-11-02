# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Homework on two-way fixed effects
# 
# *29 September, 2021*
# 
# The goal of the following homework is to develop our understanding of two-way fixed effect models. 
# 
# Related papers:
#  - the original paper by [Abowd, Kramarz, and Margolis](https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00020).
#  - [Andrews et al paper](https://www.jstor.org/stable/30135090)
#  
# %% [markdown]
# ## Ed Notes
# - k indexes firm/firm type
# - l indexes worker/worker type
# %% [markdown]
# ## Preparing the environment

# %%
using Pkg
Pkg.activate(".") # Create new environment in this folder

# first time you need to install dependencies
# Pkg.add("Distributions")
# Pkg.add("StatsBase")
# Pkg.add(["DataFrames","DataFramesMeta","Chain"])
# Pkg.add("Plots")
# Pkg.add("CategoricalArrays")
# Pkg.add("LightGraphs")
# Pkg.add("Optim")
# Pkg.add("FixedEffectModels")
# Pkg.add("Flux")
# Pkg.add("ShiftedArrays")

# past the first time, you only need to instanciate the current folder
# Pkg.instantiate(); # Updates packages given .toml file

# %% [markdown]
# We then list our imports

# %%
using Distributions
using LinearAlgebra
using StatsBase
using DataFrames
using Plots
using CategoricalArrays
using ShiftedArrays
using Chain
using DataFramesMeta
# using LightGraphs
# using TikzGraphs
using Optim
using Random

# %% [markdown]
# ## Constructing Employer-Employee matched data
# %% [markdown]
# ### Create a mobility matrix
# 
# One central piece is to have a network of workers and firms over time. We start by simulating such an object. The rest of the homework will focus on adding wages to this model. As we know from the lectures, a central issue of the network will be the number of movers.
# 
# We are going to model the mobility between workers and firms. Given a transition matrix we can solve for a stationary distribution, and then construct our panel from there.

# %%
struct define_parameters
    v_sd::Float64 # Firm amenity SD
    α_sd::Float64 # worker type sd
    ψ_sd::Float64 # Firm SD
    vψ_sd::Float64 # Cov(Amenity, Wage) 
    cnetw::Float64 # network
    csig::Float64 # cross sectional sd
end

struct define_hyper_parameters
    nk::Int # firm types
    nl::Int # worker types
    nt::Int # time period
    ni::Int # n indiv
    λ::Float64 # moving prob
    δ::Float64 # death prob
    fpt::Float64
end

struct compute_transition_matrix
    ψ::Vector
    G::Array
    H::Array
    v::Array
    α::Vector


    function compute_transition_matrix(parameters::define_parameters, hyper_parameters::define_hyper_parameters)

        # setting parameters
        v_sd  = parameters.v_sd
        α_sd = parameters.α_sd
        ψ_sd  = parameters.ψ_sd 
        vψ_sd = parameters.vψ_sd
        cnetw = parameters.cnetw
        csig  = parameters.csig

        nk = hyper_parameters.nk
        nl = hyper_parameters.nl

        # approximate each distribution with some points of support
        # Firm fixed effect:
        function get_firm_characteristics(vψ_sd)
            """
            Input:
            vψ_sd: Covariance between amenities and firm FE.
            
            Output:
            ψ: Firm fixed effect
            v_a: Amenity value
            v: Total Firm value
            pr_job_offer: Probability of sending a job offer 
            """
            Σ = [v_sd vψ_sd; vψ_sd ψ_sd]
            
            Φ = rand(MvNormal([0, 0], Σ), 100000)'

            ψ = quantile(Φ[:,2], (1:nk) / (nk + 1))
            
            v_a = shuffle(quantile(Φ[:,1], (1:nk) / (nk + 1)))

            # Value of the firm (This will be fixed over time):
            v = ψ .+ v_a # this should be something like firm FE + some amenity v_a

            pr_job_offer = pdf(Normal(0, 1), shuffle(v))
            
            pr_job_offer = pr_job_offer./sum(pr_job_offer)
    
            return ψ, v_a, v, pr_job_offer
        end
        

        # Individual type fixed effect (just for the wage equation):

        function get_individual_characteristics()
            """            
            Output:
            α: individual fixed effects
            pr_death: Probability of death is random.
            """

            α = quantile.(Normal(), (1:nl) / (nl + 1))*α_sd

            pr_death = rand()
    
            return α, pr_death
        end


        ψ, v_a, v, pr_job_offer = get_firm_characteristics(vψ_sd)

        α, pr_death = get_individual_characteristics()

        # Let's create type-specific transition matrices
        # We are going to use joint normals centered on different values
        G = zeros(nl, nk, nk)

        for l in 1:nl, k in 1:nk
            # Transition probability weighted by pr of job offer (which is random)
            G[l, k, :] = pdf( Normal(0, csig), v .- cnetw * v[k])  # Get prob based on firm value
            G[l, k, :] = G[l, k, :] #.* pr_job_offer # Weight by pr of job offer
            G[l, k, :] = G[l, k, :] ./ sum(G[l, k, :]) # Normalize probability
        end

        # We then solve for the stationary distribution over psis for each alpha value
        # We apply a crude fixed point approach
        H = ones(nl, nk) ./ nk

        new(ψ, G, H, v, α)

    end

end


struct simulation_draw

    df::DataFrame
    hyper_parameters::define_hyper_parameters
    transition_matrix::compute_transition_matrix

    function simulation_draw(hyper_parameters::define_hyper_parameters, transition_matrix::compute_transition_matrix) 

        α = transition_matrix.α
        ψ = transition_matrix.ψ
        G = transition_matrix.G
        H = transition_matrix.H
        v = transition_matrix.v

        # Setting parameters
        nk = hyper_parameters.nk
        nl = hyper_parameters.nl
        ni = hyper_parameters.ni
        nt = hyper_parameters.nt
        firms_per_type = hyper_parameters.fpt

        λ  = hyper_parameters.λ
        δ  = hyper_parameters.δ
        w_sigma = 0.2
        

        # We simulate a balanced panel
        ll = zeros(Int64, ni, nt) # Worker type
        kk = zeros(Int64, ni, nt) # Firm type
        id = zeros(Int64, ni, nt) # Individual ID
        spellcount = zeros(Int64, ni, nt) # Employment spell

        id_count = 0

        for i in 1:ni   # Iterate over individuals

            id_count += 1
            
            # At time 1, we draw random type.
            l = rand(1:nl)
            ll[i,1] = l   # N x T
            
            # At time 1, we draw random firm, no stationary dist.
            kk[i,1] = sample(1:nk)

            # Insert id of the person:
            id[i,1] = id_count

            # Now iterate over time to figure where they are going
            for t in 2:nt

                kk_past = kk[i,t-1]

                if rand() > δ  # If doesn't die.

                    id[i,t] = id[i,t-1]   #keep the same id
                    ll[i,t] = l          #keep same worker type

                    if rand() < λ  #Consider λ which determines probability of receiving an offer

                        # Here person leaves...
                        kk_new = sample(1:nk, Weights(G[l, kk_past, :]))     # Get random firm type that makes the offer pr. G | kk[i,t-1] and weighted by vacancy rate.

                        if v[kk_past] < v[kk_new] + rand(Gumbel(0, 1))

                            kk[i,t] = kk_new                                    # Otherwise, remain at same job
                            spellcount[i,t] = spellcount[i,t-1] + 1             # Add 1 to spell count
                        
                        else
                            
                            kk[i,t] = kk_past                                   # Otherwise, remain at same job
                            spellcount[i,t] = spellcount[i,t-1]                 # person didn't like the firm too much so he stays
                        
                        end

                    else

                        kk[i,t] = kk_past                                       # Otherwise, remain at same job
                        spellcount[i,t] = spellcount[i,t-1]                     # Do not add to spell count

                    end
                
                else    # if dies

                    # Fill spot with new guy and guy id 
                    id_count += 1       
                    id[i,t] = id_count  

                    #Draw new type for the guy
                    l = rand(1:nl)      
                    ll[i,t] = l        

                    # The worker starts at the same firm as the previous guy:
                    kk[i,t] = kk_past

                    # [TODO] not sure if this spellcount is correct
                    spellcount[i,t] = 1                     

                end

            end
            
        end

        #------------------------------------------------------------

        # This part likely to be the same:
        
        jj = zeros(Int64, ni, nt) # Firm identifiers

        draw_firm_from_type(k) = sample(1:firms_per_type) + (k - 1) * firms_per_type  # This samples firm code conditional on type. 

        for i in 1:ni
            
            # Extract firm type
            k = kk[i,1]
            
            # We draw the firm (one of firms_per_type in given group)
            jj[i,1] = draw_firm_from_type(k)
            
            for t in 2:nt
                if spellcount[i,t] == spellcount[i,t-1]
                    # We keep the firm the same
                    jj[i,t] = jj[i,t-1]
                else
                    # We draw a new firm
                    k = kk[i,t]
                    
                    new_j = draw_firm_from_type(k)            
                    # Make sure the new firm is actually new
                    while new_j == jj[i,t-1]
                        new_j = draw_firm_from_type(k)
                    end
                    
                    jj[i,t] = new_j
                end

            end

        end
        # Make sure firm ids are contiguous
        contiguous_ids = Dict( unique(jj) .=> 1:length(unique(jj))  )
        jj .= getindex.(Ref(contiguous_ids),jj);

        #----------------------------------------------------------------------------------------
        # TODO ii is now incorrect as dead individuals are counted as the same 
        # their successors.
        ii = repeat(1:ni,1,nt)
        tt = repeat((1:nt)',ni,1)
        df = DataFrame(i=ii[:], j=jj[:], l=ll[:], k=kk[:], α=α[ll[:]], ψ=ψ[kk[:]], t=tt[:], w_id=id[:] , spell=spellcount[:]);
        
        df[!, :lw] = df.α + df.ψ + w_sigma * rand(Normal(), size(df)[1]);

        new(df, hyper_parameters, transition_matrix)
    
    end

end


# Get the next firm the mover is moving to
# by leading the firm column and extracting 
# just the firm and next firm as one observation
function find_firm_links(mover_df::DataFrame)
    firm_link_df = @chain mover_df begin
        sort([:w_id, :t])
        groupby(:w_id)
        transform(:j => lead => :j_next)
        transform([:j, :j_next] => ByRow(==) => :j_j_next_BroadcastFunction)
        subset(:j_j_next_BroadcastFunction => x -> x .== false, skipmissing = true)
        select(:j, :j_next)
    end
    return firm_link_df
end

function create_M_flows(link_df)
    M = @chain link_df begin
        groupby([:j, :j_next])
        combine(nrow => :count, [:j, :j_next] => ((x, y) -> ("M_" .* string.(y) .*"_" .* string.(x))) => :M)
    end
    return M
end


# %%


function create_adjacency_matrix(firm_link_df::DataFrame, df::DataFrame; count = false)
    adjacency_matrix = zeros(Int, maximum(df.j), maximum(df.j))
    if count == false
        firm_link_df[!, "count"] .= 1
    end
    for firm_a in unique(df.j), firm_b in unique(df.j)
        subset_a_df = firm_link_df[(firm_link_df.j .== firm_a) .& (firm_link_df.j_next .== firm_b), :]
        if size(subset_a_df)[1] != 0
            adjacency_matrix[firm_b, firm_a] = subset_a_df.count[1]
        end
    end
    return adjacency_matrix
end

# Find all movers defined as those with max spell > 0
function find_movers(df::DataFrame)
    move_df = @chain df begin
       groupby(:w_id)
       transform(:spell => maximum) 
       subset(:spell_maximum => x -> x .> 0)
    end

   return move_df 
end

function create_fixed_point_matrices(M_df, df)
    sum_flows = @chain M_df begin
        groupby(:j)
        combine(:count => sum)
        sort(:j)
    end
    S_kk = Diagonal(sum_flows.count_sum)
    M_0 = create_adjacency_matrix(M_df, df, count = true)
    return S_kk, M_0
end


function create_fixed_point_matrices(df)
    link_df = find_firm_links(find_movers(df))
    M_df = create_M_flows(link_df)

    sum_flows = @chain M_df begin
        groupby(:j)
        combine(:count => sum)
        sort(:j)
    end
    S_kk = Diagonal(sum_flows.count_sum)
    M_0 = create_adjacency_matrix(M_df, df, count = true)
    return S_kk, M_0
end



function estimate_rank(S_kk, M_0; tol= 1e-10)
    N_connected = size(S_kk, 1)
    S_inv = inv(S_kk)
    initial_V_EE = fill(1.0, N_connected)
    lhs = S_inv * M_0 * exp.(initial_V_EE)
    rhs = exp.(initial_V_EE)
    I_M = Matrix(1I, size(M_0))
    i = 0
    max_diff = Inf
    while max_diff > tol
        i += 1
        rhs = lhs
        lhs = S_inv*M_0* rhs
        max_diff = maximum(abs.(lhs .- rhs))
        println("max diff: $max_diff")
    end
    println("Iters: $i")
    return log.(rhs)
end


# %% 
# Execute data generating process
sim_parameters = define_parameters(1.0, 1.0, 1.0, 0.5, 0.2, 0.2)
sim_hyper_parameters = define_hyper_parameters(5, 10, 10, 10_000, 0.4, 0.1, 3)
sim_transition_matrix = compute_transition_matrix(sim_parameters, sim_hyper_parameters)

size(sim_transition_matrix.G)
size(sim_transition_matrix.H)

# %%
sim_draw = simulation_draw(sim_hyper_parameters, sim_transition_matrix)
sim_df = sim_draw.df


# %% 

link_df = find_firm_links(find_movers(sim_df))

M_df = create_M_flows(link_df)

S_kk, M_0 = create_fixed_point_matrices(sim_df)
S_kk= Diagonal(reshape(sum(M_0, dims=1), (15,)))


rank = estimate_rank(S_kk, M_0)

hcat(inv(S_kk) * M_0 * exp.(rank), exp.(rank))



# sort(sim_df, :w_id)
# unique(sim_df[:,:w_id])



# %% [markdown]
# And we can plot the joint distribution of matches

# %%
if show_plots == true
    plot(sim_transition_matrix.H, xlabel="Worker", ylabel="Firm", zlabel="H", st=:wireframe)
end


# %% [markdown]
# <span style="color:green">Question 1</span>
# 
#  - Explain what the parameters `cnetw` and  `csort` do.
# 
# <span style="color:aqua">
# They create heterogeneous network and sorting effects for each firm/worker type's transition matrix by changing the density the pdf evaluates to calculate the transition matrix.
# 
# </span>
# %% [markdown]
# ### Simulate a panel
# 
# The next step is to simulate our network given our transition rules.


# # %% [markdown]
# # ### Attach firm ids to types
# %% [markdown]
# <span style="color:green">Question 2</span>
# 
#  - Explain the last 2 lines, in particular `.=>` and the use of `Ref`. 
# 
# <span style="color:aqua">
# 
# The initial for-loop in the cell above randomly samples a firm, within each firm type. Suppose there's only one individual and one firm type and five time periods, the sampled firm IDs could be:
# 3, 3, 4, 4, 4. It would be a bit silly to have firm IDs 3 and 4 with only two firms present in the entire dataset.
# 
# 
# Therefore, the last two lines remap the randomly sampled firm IDs to a new unique firm ID between 1 and the maximum number of firms. That is, ensuring firm ID is contiguous so we don't have 100 firms but a firm with ID 256.  
# 
# 
# 
# `.=>` creates a pair which is a bit like a `{key:value}` relationship mapping the original IDs to their new contiguous value. The final line remaps the matrix of original firm IDs to their new contiguous firm IDs.
# 
# </span>
# 
# 
# 
# 


# %% [markdown]
# <span style="color:green">Question 3</span>
# 
# Use `Chain.jl` and `DataFramesMeta.jl` to computer:
# 
#  - mean firm size, in the crossection, expect something like 15.
#  - mean number of movers per firm in total in our panel.
# 

# %%

last(sim_df, 500)


# %%
@chain sim_df begin
    groupby([:j, :t])
    combine(nrow => :count)
    groupby(:j)
    combine(:count => mean)
    @aside println("On average each firm has $(round(mean(_.count_mean))) observations, averaging across time periods.")
    first(_, 5)
end


# %%
#Number of movers at the firm, defined as people that moved into the firm at any point in time:
#Here we can be double (or more) counting returners as new movers for the firm  ...

# N.B. this doesn't work anymore as spell count changes when someone dies.
moversDataFrame = @chain sim_df begin
    groupby([:j, :w_id, :spell]) # Group at the firm, individual, spell level
    combine(:t => mean)       # Collapse, this column doesn't matter actually
    @transform!(:spell_true = :spell .> 0) # If spell != 0 it means individual moved in (don't care when or if returner)
    groupby(:j) # Group by firm
    combine(:spell_true => sum) # Count number of movers
    @aside println("On average each firm has $(round(mean(_.spell_true_sum))) movers each period.")
end


# %% [markdown]
# ## Simulating AKM wages and create Event Study plot
# %% [markdown]
# We start with just AKM wages, which is log additive with some noise.

# %%

# %% [markdown]
# <span style="color:green">Question 4</span>
# 
# Before we finish with the simulation code. Use this generated data to create the event study plot from [Card. Heining, and Kline](https://doi.org/10.1093/qje/qjt006):
# 
# 1. Compute the mean wage within firm
# 2. Group firms into quartiles
# 3. Select workers around a move (2 periods pre, 2 periods post)
# 4. Compute wages before/after the move for each transition (from each quartile to each quartile)
# 5. Plot the lines associated with each transition

# %%
sim_df[!, :dummy] .= 1;


#1. Mean wage within firm: 
function find_move_year(df, order, sort_id_1, sort_id_2, spell_var)
    if order == "forward"
        order = true
        varname = :years_at_firm_sofar
    end

    if order == "backward" 
        order = false
        varname = :years_at_firm_sofar_inverse
    end
    

    year_df =  @chain df begin
        sort([sort_id_1, sort_id_2], rev = order)
        groupby([sort_id_1, spell_var])
        transform(:dummy .=> cumsum => varname)
    end
    return  year_df
end

# %%


eventStudyPanel =  @chain sim_df begin
    groupby([:j, :t]) # Group by firm and time 
    transform(:lw => mean) # Obtain wage at that group level
    @transform!(:wage_percentile = cut(:lw_mean, 4)) # Get wage_percentile at same level of agregation
    groupby([:w_id, :spell])
    transform(nrow => :years_at_firm) # Get number of years individual spend at a single firm
    find_move_year(_, "forward", :w_id, :t, :spell) # generating event time indicators
    find_move_year(_, "backward", :w_id, :t, :spell)


    @aside initialFirmDataFrame = @chain _ begin
        subset(:spell => ByRow(==(1)), :years_at_firm_sofar_inverse => ByRow(<=(2))) # Generate dataframe for initial firm, keep last two years (call it initialFirmDataFrame)
    end

    subset(:spell => ByRow(==(0)), :years_at_firm_sofar => ByRow(<=(2))) # Generate dataframe for subsequent firm, keep first two years
    append!(initialFirmDataFrame) # Append both dataframes
    groupby(:w_id) # group by person 
    transform(nrow => :nreps) # and get number of repetitions
    subset(:nreps => ByRow(==(4))) # and get number of repetitions
    sort([:w_id,:t]) # sort by i and t to check if it worked out
    # Generating event time variable:
    sort([:w_id,:t], rev = false) # Sort over time by individual (this helps to get first 2 periods of last firm)
    groupby([:w_id])  # by worker and time
    transform(:dummy .=>  cumsum => :event_time) # Get the number of years spent at a firm so far, but backwards.
end

# Define the dictionary with quartile labels to refer when filtering:
percentile_cut = Dict(1:4 .=> unique(sort(eventStudyPanel.wage_percentile)))


# %%
# Get Card Heining and Kling event Figure:
# There may be a smart way to do this, here I'm brute forcing it ...
initial = 1
final = 4
function generateEventStudy(eventStudyPanel, initial, final)
    return  eventStudy =  @chain eventStudyPanel begin
                            @aside finalJob = @chain _ begin
                                subset(:wage_percentile => ByRow(==(percentile_cut[final])), :spell => ByRow(==(1)))
                            end
                            subset(:wage_percentile => ByRow(==(percentile_cut[initial])), :spell => ByRow(==(0)))
                            append!(finalJob) # Append both dataframes
                            groupby(:w_id)
                            transform(nrow => :nreps)
                            subset(:nreps => ByRow(==(4)))
                            groupby([:wage_percentile, :event_time])
                            combine(:lw => mean)
                            sort(:event_time)
    end
    return eventStudy
end


eventStudySwitchersDown = broadcast(
    x -> generateEventStudy(eventStudyPanel, 1, x).lw_mean, 
    [1, 2, 3, 4]
)
eventStudySwitchersUp = broadcast(
    x -> generateEventStudy(eventStudyPanel, 4, x).lw_mean,
    [1, 2, 3, 4]
)
plot(1:4,eventStudySwitchersDown, label=["1 to 1" "1 to 2" "1 to 3" "1 to 4"],markershape = :square)
plot!(1:4,eventStudySwitchersUp, label =  ["4 to 1 " "4 to 2" "4 to 3" "4 to 4"],markershape = :circle)
plot!(legend=:outertopright)

# %% [markdown]
# ## Calibrating the parameters
# %% [markdown]
# <span style="color:green">Question 5</span>
# 
#  - Pick the parameters `psi_sd`, `alpha_sd`, `csort`, `csig`, and `w_sigma` to roughly match the decomposition in the Card-Heining-Kline paper (note that they often report numbers in standard deviations, not in variances).

# %%
# Generate function for data generating process, including as arguments parameters
# we want to calibrate:

# %%
# Function to compute variance decomposition...

function variance_decomposition(df, true_parameters=true)

    if true_parameters == true
        sig_α = var(df.α)
        sig_ψ = var(df.ψ)
        sig_αψ = 2*cov(df.α, df.ψ)
        cor_αψ = cor(df.α, df.ψ)
        sig_lw = var(df.lw)


    else
        sig_α = var(df.α_hat)
        sig_ψ = var(df.ψ_hat)
        sig_αψ = 2*cov(df.α_hat, df.ψ_hat)
        sig_lw = var(df.lw)
    end


    return [sig_α, sig_ψ, sig_αψ, cor_αψ, sig_lw]

end

function variance_calibration(parameters::define_parameters,
                              hyper_parameters::define_hyper_parameters)
    transition_mat = compute_transition_matrix(parameters, hyper_parameters)
    sim_draw = simulation_draw(hyper_parameters, transition_mat)

    return variance_decomposition(sim_draw.df, true)

end


# %% 
function anon_function(param_list)
    # parameters(v_sd, \alpha_sd, ψ_sd, vψ_sd, cnetw, csig)
    parameters = parameters(param_list[1], param_list[2], param_list[3], 0.2, 0.2)
    values = variance_calibration(parameters, sim_hyper_parameters)
    target_values = [
        0.51, # variance of person effect 
        0.14, # variance of employer effect
        0.10, # 2 cov(person, employer)
        0.19, # corr(person, employer)
        0.67, # variance log earnings

        0.400, # V^{EE} and \psi (pearson)
        0.530, # V^e and \psi (pearson)
        0.045, # V^{EE} log(size) (pearson)
        0.151, # V^e log(size)(pearson)
        0.093  # ψ log size (pearson)
    ]
    target = [0.084, 0.025, 0.003, 0.138]
    mse = mean((target .- values).^2)
    return mse
end

# %%|
using Optim
results = Optim.optimize(
    anon_function,
    [0.25, 0.25, 0.25, 0.25],
    # LBFGS(),
    NelderMead(),
    Optim.Options(iterations = 10_000)
)

# %%
# %%
using Optim
approx_results = Optim.optimize(
    anon_function,
    [0.25, 0.25, 0.25, 0.25],
    # LBFGS(),
    NelderMead(),
    Optim.Options(g_abstol = 1e-6)
)

approx_calibrated_parameters = parameters(
    Optim.minimizer(approx_results)[1],
    Optim.minimizer(approx_results)[2],
    Optim.minimizer(approx_results)[3],
    0.2,
    Optim.minimizer(approx_results)[4],
    0.2
    # Optim.minimizer(results)[5] 
)


# %% [markdown]
# The optimisation function literally returns whatever the initial parameters are but still
# returns a success flag... nice. We <3 identification.
#


# %%
calibrated_transition_matrix  = compute_transition_matrix(calibrated_parameters, sim_hyper_parameters)
calibrated_sim = simulation_draw(sim_hyper_parameters, calibrated_transition_matrix)

values = variance_calibration(calibrated_parameters, sim_hyper_parameters)
target = [0.084, 0.025, 0.003, 0.137]
mse = mean((target .- values).^2)
# %% [markdown]
# Generating calibrated plots - these don't look totally convincing
# %%

p1 = plot(calibrated_transition_matrix.G[1, :, :], xlabel="Previous Firm", ylabel="Next Firm", zlabel="G[1, :, :]", st=:wireframe)
p2 = plot(calibrated_transition_matrix.G[end, :, :], xlabel="Previous Firm", ylabel="Next Firm", zlabel="G[nl, :, :]", st=:wireframe, right_margin = 10Plots.mm) # right_margin makes sure the figure isn't cut off on the right
plot(p1, p2, layout = (1, 2), size=[600,300])
# %%
plot(calibrated_transition_matrix.H, xlabel="Worker", ylabel="Firm", zlabel="H", st=:wireframe)


# %% [markdown]
# ## Getting connected set:




# %%
##### Ed attempt 

# Iterate through our firm link df and create a matrix 
# of links. In R I would model.matrix(a ~ b) but idk how to 
# do that here
function create_adjacency_matrix(firm_link_df::DataFrame, df::DataFrame)
    adjacency_matrix = zeros(Int, maximum(df.j), maximum(df.j))
    for firm_a in unique(df.j), firm_b in unique(df.j)
        subset_a_df = firm_link_df[(firm_link_df.j .== firm_a) .& (firm_link_df.j_next .== firm_b), :]
        if size(subset_a_df)[1] != 0
            adjacency_matrix[firm_a, firm_b] = 1
            adjacency_matrix[firm_b, firm_a] = 1
        end
    end
    return adjacency_matrix
end


function create_connected_df(df::DataFrame)
    move_df = find_movers(df)
    firm_df = find_firm_links(move_df)
    adjacency_matrix = create_adjacency_matrix(firm_df, move_df)


    simple_graph = SimpleGraph(adjacency_matrix);

    connected_network = connected_components(simple_graph)

    connected_set = connected_network[1]

    println("We have only $(length(connected_set)) firms fully connected $(length(connected_set)/maximum(df.j)) of the market...") # Double check we might be wrong...

    df_connected = df[in(connected_set).(df.j),:]
    return df_connected
end


# %%
df_connected = create_connected_df(sim_df)



# %% AKM Estimation:

function akm_estimation(df::DataFrame)
    df[!, :α_hat] .= 0
    df[!, :ψ_hat] .= 0

    delta = Inf 
    tol = 1e-5
    transform!(x -> mean(x.lw - x.ψ_hat), groupby(df, :i))    
    df.α_hat = df.x1
    transform!(x -> mean(x.lw - x.α_hat), groupby(df, :j))
    df.ψ_hat = df.x1
    mse = mean((df.lw .- df.α_hat .- df.ψ_hat).^2)


    while delta > tol

        transform!(x -> mean(x.lw - x.ψ_hat), groupby(df, :i))    
        df.α_hat = df.x1
        transform!(x -> mean(x.lw - x.α_hat), groupby(df, :j))
        df.ψ_hat = df.x1
        new_mse = mean((df.lw .- df.α_hat .- df.ψ_hat).^2)
        delta =  new_mse - mse
        mse = new_mse
    end
    return df
end


# %%
df_connected_results = akm_estimation(df_connected);
# %%

# Q8: Limited mobility bias
#---------------------------------------------------------------------
# %%

function run_mob_bias_simulation(λ::Float64, nt::Int, parameters::parameters)
        λ_hyper_parameters = hyper_parameters(30, 10, λ, nt, 10_000) # gen data changing λ and nt
        
        λ_transition_matrix = compute_transition_matrix(parameters, λ_hyper_parameters)
        
        λ_sim = simulation_draw(λ_hyper_parameters, λ_transition_matrix)
        λ_df = λ_sim.df
        λ_df_connected = create_connected_df(λ_df) 

        λ_df_connected_results = akm_estimation(λ_df_connected)

        cov_αψ = cov(λ_df_connected_results.α, λ_df_connected_results.ψ)
        cov_αψ_hat = cov(λ_df_connected_results.α_hat,λ_df_connected_results.ψ_hat )

        cov_bias = cov_αψ - cov_αψ_hat
        var_ψ = var(λ_df_connected_results.ψ)
        var_ψ_hat = var(λ_df_connected_results.ψ_hat)
        var_bias = var_ψ - var_ψ_hat
        result_df = DataFrame(
            Dict(
                :cov_αψ => cov_αψ,
                :cov_αψ_hat => cov_αψ_hat, 
                :cov_bias => cov_bias, 
                :var_ψ => var_ψ,
                :var_ψ_hat => var_ψ_hat,
                :var_bias => var_bias,
                :T => nt,
                :λ => λ
            )
        )


        return  result_df
end



# %%


λ_list = [0.1, 0.3, 0.5, 0.7, 0.9]
nt_list = [5, 6, 8, 10, 15];



λ_bias_df = broadcast(
    x -> run_mob_bias_simulation(x, nt_list[2], sim_parameters),
    λ_list
)

# %%

λ_bias_plot_df = reduce(vcat, λ_bias_df)
plot(
    λ_bias_plot_df[!, :λ],
     λ_bias_plot_df[!, :cov_αψ],
     markershape= :circle,
     linetype = :scatter)
plot!(
    λ_bias_plot_df[!, :λ],
    λ_bias_plot_df[!, :cov_αψ_hat])



# %%
plot(
   λ_bias_plot_df.λ,
   λ_bias_plot_df.var_bias
)



# %%

nt_bias_df = broadcast(
    x -> run_mob_bias_simulation(λ_list[1], x, sim_parameters),
    nt_list
)

nt_bias_df = reduce(vcat, nt_bias_df)


# %%

plot(
    nt_bias_df[!, :T],
     nt_bias_df[!, :cov_αψ],
     markershape= :circle,
     linetype = :scatter)
plot!(
    nt_bias_df[!, :T],
    nt_bias_df[!, :cov_αψ_hat])



# %%
plot(
   nt_bias_df.T,
   nt_bias_df.var_bias
)

# 
# %% [markdown]
# ## Limited mobility bias
# %% [markdown]
# We now have everything we need to look at the impact of limited mobility bias. Compute the following:
# 
# 1. Compute the estimated variance of firm FE
# 2. Do it for varying levels of mobility λ. For each the number of movers, collect the actual variance and the estimated variance. Run it for different panel lengths: 5, 6, 8, 10, 15.
# 
# <span style="color:green">Question 8</span>
# 
#  - Report this in a plot. This should look like the [Andrews et al.](https://www.sciencedirect.com/science/article/pii/S0165176512004272) plot.
# %% [markdown]
# ## Correcting the bias
# %% [markdown]
# <span style="color:green">Question 9</span>
# 
#  - Implement both the exact as well as the approximated (Hutchkinson trace approximation) Andrews et al. type fixed-effect bias correction based on homoskedacity. Redo the plot from Question 6 so that it includes these 2 estimates.
# %% [markdown]
# ## Evidence of learning
# 
# <span style="color:green">Question 10</span>
# 
#  - Regress the wage of a worker at time $t$ on his wage at time $t-l$ and the average wage of his co-workers at time $t-l$ for some lags $l$ focusing on workers that did change firms between $t-l$ and $t$. 
#  - Comment on the result.
# 

