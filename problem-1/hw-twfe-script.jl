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
Pkg.add("Distributions")
Pkg.add("StatsBase")
Pkg.add(["DataFrames","DataFramesMeta","Chain"])
Pkg.add("Plots")
Pkg.add("CategoricalArrays")
Pkg.add("LightGraphs")
Pkg.add("FixedEffectModels")

# past the first time, you only need to instanciate the current folder
Pkg.instantiate(); # Updates packages given .toml file

# %% [markdown]
# We then list our imports

# %%
using Distributions
using LinearAlgebra
using StatsBase
using DataFrames
using Plots
using CategoricalArrays
using FixedEffectModels
using Chain
using DataFramesMeta
using LightGraphs
using TikzGraphs
using Optim

# %% [markdown]
# ## Constructing Employer-Employee matched data
# %% [markdown]
# ### Create a mobility matrix
# 
# One central piece is to have a network of workers and firms over time. We start by simulating such an object. The rest of the homework will focus on adding wages to this model. As we know from the lectures, a central issue of the network will be the number of movers.
# 
# We are going to model the mobility between workers and firms. Given a transition matrix we can solve for a stationary distribution, and then construct our panel from there.

# %%
struct parameters
    α_sd::Float64 # indiv std
    ψ_sd::Float64 # firm sd
    csort::Float64 # sorting
    cnetw::Float64 # network
    csig::Float64 # cross sectional sd
    w_sigma::Float64
end

struct hyper_parameters
    nk::Int # firm types
    nl::Int # worker types
    λ::Float64 # moving prob
    nt::Int # time period
    ni::Int # n indiv
end

# %%

function gen_transition_matrix(params::parameters, hyper_params::hyper_parameters)
    # setting params
    α_sd = params.α_sd
    ψ_sd = params.ψ_sd 
    csort = params.csort
    cnetw = params.cnetw
    csig = params.csig

    nk = hyper_params.nk
    nl = hyper_params.nl


    # approximate each distribution with some points of support
    ψ = quantile.(Normal(), (1:nk) / (nk + 1)) * ψ_sd
    α = quantile.(Normal(), (1:nl) / (nl + 1)) * α_sd

    # Let's create type-specific transition matrices
    # We are going to use joint normals centered on different values
    G = zeros(nl, nk, nk)
    for l in 1:nl, k in 1:nk
        G[l, k, :] = pdf( Normal(0, csig), ψ .- cnetw * ψ[k] .- csort * α[l])
        G[l, k, :] = G[l, k, :] ./ sum(G[l, k, :])
    end

    # We then solve for the stationary distribution over psis for each alpha value
    # We apply a crude fixed point approach
    H = ones(nl, nk) ./ nk
    for l in 1:nl
        M = transpose(G[l, :, :])
        for i in 1:100
            H[l, :] = M * H[l, :]
        end
    end
    return α, ψ, G, H
end



function gen_dataset(
    hyper_params::hyper_parameters,
    α::Array{Float64, 1},
    ψ::Array{Float64, 1},
    G::Array{Float64, 3},
    H::Array{Float64, 2}
)
    # setting params
    nk = hyper_params.nk
    nl = hyper_params.nl
    ni = hyper_params.ni
    nt = hyper_params.nt
    λ = hyper_params.λ
    w_sigma = 0.2

    # We simulate a balanced panel
    ll = zeros(Int64, ni, nt) # Worker type
    kk = zeros(Int64, ni, nt) # Firm type
    spellcount = zeros(Int64, ni, nt) # Employment spell

    for i in 1:ni
        
        # We draw the worker type
        l = rand(1:nl)
        ll[i,:] .= l
        
        # At time 1, we draw from H
        kk[i,1] = sample(1:nk, Weights(H[l, :]))
        
        for t in 2:nt
            if rand() < λ
                kk[i,t] = sample(1:nk, Weights(G[l, kk[i,t-1], :]))
                spellcount[i,t] = spellcount[i,t-1] + 1
            else
                kk[i,t] = kk[i,t-1]
                spellcount[i,t] = spellcount[i,t-1]
            end
        end
        
    end

    #------------------------------------------------------------

    firms_per_type = 15
    jj = zeros(Int64, ni, nt) # Firm identifiers

    draw_firm_from_type(k) = sample(1:firms_per_type) + (k - 1) * firms_per_type

    for i in 1:ni
        
        # extract firm type
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

    ii = repeat(1:ni,1,nt)
    tt = repeat((1:nt)',ni,1)
    df = DataFrame(i=ii[:], j=jj[:], l=ll[:], k=kk[:], α=α[ll[:]], ψ=ψ[kk[:]], t=tt[:], spell=spellcount[:]);
    
    df[!, :lw] = df.α + df.ψ + w_sigma * rand(Normal(), size(df)[1]);


    return df
    
end


# %%


# %%

initial_params = parameters(1.0, 1.0, 0.5, 0.2, 0.5, 0.2)
initial_hyper_params = hyper_parameters(30, 10, 0.1, 10, 10_000)
α, ψ, G, H = gen_transition_matrix(initial_params, initial_hyper_params)


# # %%


# p1 = plot(G[1, :, :], xlabel="Previous Firm", ylabel="Next Firm", zlabel="G[1, :, :]", st=:wireframe)
# p2 = plot(G[initial_hyper_params.nl, :, :], xlabel="Previous Firm", ylabel="Next Firm", zlabel="G[nl, :, :]", st=:wireframe, right_margin = 10Plots.mm) # right_margin makes sure the figure isn't cut off on the right
# plot(p1, p2, layout = (1, 2), size=[600,300])

# %%

df = gen_dataset(
    initial_hyper_params,
    α,
    ψ,
    G,
    H
)

p1 = plot(G[1, :, :], xlabel="Previous Firm", ylabel="Next Firm", zlabel="G[1, :, :]", st=:wireframe)
p2 = plot(G[end, :, :], xlabel="Previous Firm", ylabel="Next Firm", zlabel="G[nl, :, :]", st=:wireframe, right_margin = 10Plots.mm) # right_margin makes sure the figure isn't cut off on the right
plot(p1, p2, layout = (1, 2), size=[600,300])

# %% [markdown]
# And we can plot the joint distribution of matches

# %%
plot(H, xlabel="Worker", ylabel="Firm", zlabel="H", st=:wireframe)

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

last(df, 500)


# %%
@chain df begin
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

moversDataFrame = @chain df begin
    groupby([:j, :i, :spell]) # Group at the firm, individual, spell level
    combine(:t => mean)       # Collapse, this column doesn't matter actually
    @transform!(:spell_true = :spell .> 0) # If spell != 0 it means individual moved in (don't care when or if returner)
    groupby(:j) # Group by firm
    combine(:spell_true => sum) # Count number of movers
    @aside println("On average each firm has $(round(mean(_.spell_true_sum))) movers each period.")
end

# %% [markdown]
# 

# %%
@chain df begin
    groupby([:j, :i, :spell])
    transform(nrow => :spell_count)
    subset(:spell_count => ByRow(spell_count -> spell_count == 0) )
    # subset(:count => ByRow(count -> count < 10))
    # subset(:i => ByRow(i -> i == 2))
end
    


# @chain df begin
#     sort([:i, :t])
#     groupby(:i)
#     transform(:j => lag => :lag_ed)
# end


# end_id = 
# subset(df, :t => ByRow(t -> t == 10))

# @chain df begin
#     subset(df, :j => ByRow(j -> j == 1))
#     subset(df, :i => ByRow(i -> i == 1))
#     first(_, 5)
# end
# first(@subset(df, :i => ByRow(i -> i == 1)))
# subset(df, :i => ByRow(i -> i == 1))

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
df[!, :dummy] .= 1;


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




eventStudyPanel =  @chain df begin
    groupby([:j, :t]) # Group by firm and time 
    transform(:lw => mean) # Obtain wage at that group level
    @transform!(:wage_percentile = cut(:lw_mean, 4)) # Get wage_percentile at same level of agregation
    groupby([:i, :spell])
    transform(nrow => :years_at_firm) # Get number of years individual spend at a single firm
    find_move_year(_, "forward", :i, :t, :spell) # generating event time indicators
    find_move_year(_, "backward", :i, :t, :spell)


    @aside initialFirmDataFrame = @chain _ begin
        subset(:spell => ByRow(==(1)), :years_at_firm_sofar_inverse => ByRow(<=(2))) # Generate dataframe for initial firm, keep last two years (call it initialFirmDataFrame)
    end

    subset(:spell => ByRow(==(0)), :years_at_firm_sofar => ByRow(<=(2))) # Generate dataframe for subsequent firm, keep first two years
    append!(initialFirmDataFrame) # Append both dataframes
    groupby(:i) # group by person 
    transform(nrow => :nreps) # and get number of repetitions
    subset(:nreps => ByRow(==(4))) # and get number of repetitions
    sort([:i,:t]) # sort by i and t to check if it worked out
    # Generating event time variable:
    sort([:i,:t], rev = false) # Sort over time by individual (this helps to get first 2 periods of last firm)
    groupby([:i])  # by worker and time
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
                            groupby(:i)
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
plot!(1:4,eventStudySwitchersUp,label =  ["4 to 1 " "4 to 2" "4 to 3" "4 to 4"],markershape = :circle)
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
        sig_lw = var(df.lw)
    else
        sig_α = var(df.α_hat)
        sig_ψ = var(df.ψ_hat)
        sig_αψ = 2*cov(df.α_hat, df.ψ_hat)
        sig_lw = var(df.lw)
    end


    return [sig_α, sig_ψ, sig_αψ, sig_lw]

function variance_calibration(params::parameters, hyper_params::hyper_parameters)
    α, ψ, G, H = gen_transition_matrix(params, hyper_params)
    
    df = gen_dataset(
        hyper_params,
        α,
        ψ,
        G,
        H
    )    

    return variance_decomposition(df, true)

end

# %%
function anon_function(param_list)
    params = parameters(param_list[1], param_list[2], param_list[3], 0.2, param_list[4], param_list[5])
    values = variance_calibration(params, initial_hyper_params)
    target = [0.084, 0.025, 0.003, 0.137]
    mse = mean((target .- values).^2)
    return mse
end

# %%
using Optim
results = Optim.optimize(
    anon_function,
    [1, 1, 0.01, 0.012, 0.015],
    LBFGS()
)

# %%
Optim.minimizer(results)
# %%




# Compute grid search where parameter lists are evenly spaced by gap
gap = 0.20
α_sd_list = 0.01:gap:1
ψ_sd_list = 0.01:gap:1
csort_list = 0.01:gap:1 # Sorting effect
csig_list  = 0.01:gap:1 # Cross-sectional standard deviation
w_sigma_list = 0.01:gap:1

flat_gridpoints(grids) = vec(collect(Iterators.product(grids...)))

grid_points = flat_gridpoints((α_sd_list, ψ_sd_list, csort_list, csig_list,w_sigma_list));

jj = 1

true_var_decomp = [0.186 0.101 0.110]
difference_outcome = []

Pkg.add("Optim")
using Optim

# don't want negative variances but neg sorting ok
lower_bound = [0, 0, -1, 0, 0]
upper_bound = [10, 10, 10, 10, 10]

function search_function(inputs)
    α_sd,ψ_sd,csort,csig, w_sigma = inputs
    _,_,_,_, _ = grid_points[1]
    df,_,_ = data_generating_process(α_sd,ψ_sd,csort,csig, w_sigma);
    var_decomp = variance_decomposition(df)
    euclidean_dist = sum((true_var_decomp-var_decomp).^2)
    return euclidean_dist
end
# search_function([0.5, 0.5, 1, 0.5, 0.5])
results = optimize(
    search_function,
    # lower_bound,
    # upper_bound,
    [0.5, 0.5, 1, 0.5, 0.5],
    Optim.Options(
        iterations = 1_000
    )
)
summary(results)
Optim.minimizer(results)



df, G, H = data_generating_process(
    Optim.minimizer(results)[1],
    Optim.minimizer(results)[2],
    Optim.minimizer(results)[3],
    Optim.minimizer(results)[4],
    Optim.minimizer(results)[5]
)


#Apply grid search, this can be optimized for sure...
for jj in 1:length(grid_points)
    α_sd,ψ_sd,csort,csig, w_sigma = grid_points[jj]
    df,_,_ = data_generating_process(α_sd,ψ_sd,csort,csig, w_sigma);
    var_decomp = variance_decomposition(df)
    euclidean_dist = sum((true_var_decomp-var_decomp).^2)
    push!(difference_outcome, euclidean_dist)
end


# %%
# Check results and tweak csig a bit...
# Get index of minimum euclidean distance estimation:
_, index_calibration = findmin(difference_outcome)
α_sd,ψ_sd,csort,csig, w_sigma = grid_points[index_calibration]
df, G, H  = data_generating_process(α_sd,ψ_sd,csort,csig, w_sigma);


p1 = plot(G[1, :, :], xlabel="Previous Firm", ylabel="Next Firm", zlabel="G[1, :, :]", st=:wireframe)
p2 = plot(G[nl, :, :], xlabel="Previous Firm", ylabel="Next Firm", zlabel="G[nl, :, :]", st=:wireframe, right_margin = 10Plots.mm) # right_margin makes sure the figure isn't cut off on the right
plot(p1, p2, layout = (1, 2), size=[600,300])

plot(H, xlabel="Worker", ylabel="Firm", zlabel="H", st=:wireframe)
unique(df.ψ)

# The output reduces cross sectional variance too much, check what happens when ↑ csig ...
df, G, H  = data_generating_process(α_sd,ψ_sd,csort, .1, w_sigma);
var_decomp = variance_decomposition(df)
euclidean_dist = sum((true_var_decomp-var_decomp).^2)
println("Euclidean distance is: $euclidean_dist")

p1 = plot(G[1, :, :], xlabel="Previous Firm", ylabel="Next Firm", zlabel="G[1, :, :]", st=:wireframe)
p2 = plot(G[nl, :, :], xlabel="Previous Firm", ylabel="Next Firm", zlabel="G[nl, :, :]", st=:wireframe, right_margin = 10Plots.mm) # right_margin makes sure the figure isn't cut off on the right
plot(p1, p2, layout = (1, 2), size=[600,300])

# It seems csig contributes importantly to covariance! Check this again and get feedback from prof.
# I'll leave it at 0.05 which is 5 times it optimal* value.

# %% [markdown]
# ## Getting connected set:

# Getting connected set:

function individualDeterministicTransitionMatrix(df,ii)

    nfirms = length(unique(df.j))
    shift_i = @chain df begin
                        subset(:i => ByRow(.==(ii)))
                        sort(:t)
                        combine(first, groupby(_,:spell))
                    end
    
    nshifts = size(shift_i)[1]
    transitionMatrix_i = zeros(Int32, nfirms, nfirms);
    
    for ii in 1:nshifts
        if ii != nshifts
            current = shift_i.j[ii]
            next = shift_i.j[ii+1]
            transitionMatrix_i[current,next] = 1
        end
    end

    return transitionMatrix_i
end


function getConnectedDataSet(df)

    # Add all shifts happening in the economy during the whole time
    nfirms = length(unique(df.j))

    totalDeterministicShifts = zeros(nfirms, nfirms);

    for ii in unique(df.i)
        totalDeterministicShifts = totalDeterministicShifts + individualDeterministicTransitionMatrix(df,ii)
    end

    adjacencyMatrix = (UpperTriangular(totalDeterministicShifts) + transpose(LowerTriangular(totalDeterministicShifts))).>1
    
    adjacencyMatrix = adjacencyMatrix + transpose(adjacencyMatrix)

    println("There are $(sum(totalDeterministicShifts)) job shifts across time ...")
    
    println("There are $(sum(adjacencyMatrix)) edges between nodes ...")

    simpleGraph = SimpleGraph(adjacencyMatrix);

    connectedNetwork = connected_components(simpleGraph)
    
    connectedSet = connectedNetwork[1]

    println("We have only $(length(connectedSet)) firms fully connected $(length(connectedSet)/nfirms) of the market...") # Double check we might be wrong...

    return df_connected = df[in(connectedSet).(df.j),:]
    
    println("Due to unconnectedness we eliminated $(size(df)[1]-size(df_connected)[1]) observations, not much")

end

# %% AKM Estimation:

function akm_estimation(df_connected)

    df_connected[:,:α_hat] .= .0;
    df_connected[:,:ψ_hat] .= .0;

    # Compute firm type fixed effects ols model: 
    delta = Inf
    tol = 0.00001
    nIter = 1 
    msePast = 0
    while delta>tol 

        if nIter == 1
            # Regress controling for industry fixed effects...
            model_j = reg(df_connected, term(:lw) ~ fe(:j), save=true);
            # Obtain residuals (which are controled by industry fe):
            df_connected[:,:α_hat] = residuals(model_j);
        else
            # Just obtain the alpha parameters (not averaged) by netting out psi_hat from prev iter.
            df_connected[:,:α_hat] = df_connected[:,:lw] - df_connected[:,:ψ_hat]
        end

        # Average fixed effects by individual:
        df_connected = transform(groupby(df_connected, :i), :α_hat => mean => :α_hat)

        # Net out individual fixed effects
        df_connected[:,:ψ_hat] = df_connected[:,:lw] - df_connected[:,:α_hat]

        # Compute average industry fixed effects
        df_connected = transform(groupby(df_connected, :j), :ψ_hat => mean => :ψ_hat)

        # Model verification
        model_verify = reg(df_connected, @formula(lw ~ α_hat + ψ_hat), save=true);

        # Compute residuals and mse
        delta = abs(msePast - sum(residuals(model_verify, df_connected).^2)) 

        msePast = sum(residuals(model_verify, df_connected).^2) 

        # Add count
        nIter = nIter + 1

        println("At iteration number $(nIter), the MSE is $(delta)")

    end
    
    return df_connected
    
end


# %% 

# Re running the data generating process and computing akm estimation...

initial_params = parameters(1.0, 1.0, 0.5, 0.2, 0.5, 0.2)

initial_hyper_params = hyper_parameters(30, 10, 0.1, 10, 10_000)

α, ψ, G, H = gen_transition_matrix(initial_params, initial_hyper_params)

df = gen_dataset( initial_hyper_params,α,ψ,G,H)

df_connected = getConnectedDataSet(df);

df_connected_results = akm_estimation(df_connected);

# %%

# Limited mobility bias:
#---------------------------------------------------------------------

λ_list = [0.1 0.2 0.3 0.4 0.5 0.6]

nt_list = [5 6 8 10 15]

df,_,_ = data_generating_process(α_sd, ψ_sd, csort, csig, w_sigma, λ, nt);

df_connected = getConnectedDataSet(df)

df_connected_results = akm_estimation(df_connected)


# %%

# %% [markdown]
# This requires first extracting the large set of firms connected by movers, and then estimating the linear problem with many dummies.
# %% [markdown]
# ### Extracting the connected set
# %% [markdown]
# Because we are not going to deal with extremely large data-sets, we can use off-the-shelf algorithms to extract the connected set. Use the function `connected_components` from the package `LightGraphs` to extract the connected set from our data. To do so you will need to first construct an adjacency matrix between the firms. 
# 
# <span style="color:green">Question 6</span>
# 
#  - Extract the connected set and drop firms not in the set (I expect that all firms will be in the set).
# %% [markdown]
# ### Estimating worker and firm FEs
# %% [markdown]
# This part of the problem set is for you to implement the AKM estimator. As discussed in class, this can be done simply by updating, in turn, the worker FE and the firm FE.
# 
# Start by appending 2 new columns `alpha_hat` and `psi_hat` to your data. Then loop over the following:
# 
# 1. Update `alpha_hat` by taking the mean within `i` net of firm FE
# 2. Update `psi_hat` by taking the mean within `fid` net of worker FE
# 
# <span style="color:green">Question 7</span>
# 
#  - Run the previous steps in a loop, and at each step evaluate how much the total mean square error has changed. Check that is goes down with every step. Stop when the MSE decreases by less than 1e-9.
# 
# 
# Note that you can increase speed by focusing on movers only first.
# 
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

