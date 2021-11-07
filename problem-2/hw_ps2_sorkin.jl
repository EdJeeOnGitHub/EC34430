# %%
using Pkg
Pkg.activate(".") # Create new environment in this folder

<<<<<<< HEAD
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

=======
>>>>>>> c9842476bad2f8e1949a3d95931c4e62023d4134
# %% [markdown]
# We then list our imports

# %%
using Distributions
using LinearAlgebra
using Infiltrator
using StatsBase
using DataFrames
using Plots
using CategoricalArrays
using ShiftedArrays
using Chain
using DataFramesMeta
using LightGraphs
# using TikzGraphs
using Optim
using Random

include("./estimation-code.jl")
using .lewd

# %% [markdown]
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
    firms_per_type::Int
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
        firms_per_type = hyper_parameters.firms_per_type
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
            
            Φ = Normal(0,0.4)
            
            ψ = quantile.(Φ, (1:nk) / (nk + 1))
            
            v_a = quantile(Φ[:,1], (1:nk) / (nk + 1))

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
            G[l, k, k] = 0 # can't get offer from current place
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
    kk::Matrix

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
        firms_per_type = hyper_parameters.firms_per_type

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

                        
                        # For now get job offer from firms at random
                        ws = fill(1.0/(nk-1), nk)
                        ws[kk_past] = 0
                        # kk_new = sample(1:nk, Weights(G[l, kk_past, :]))     # Get random firm type that makes the offer pr. G | kk[i,t-1] and weighted by vacancy rate.
                        kk_new = sample(1:nk, Weights(ws))     # Get random firm type that makes the offer pr. G | kk[i,t-1] and weighted by vacancy rate.
                        while kk_new == kk_past
                            kk_new = sample(1:nk, Weights(G[l, kk_past, :]))     # Get random firm type that makes the offer pr. G | kk[i,t-1] and weighted by vacancy rate.
                            kk_new = sample(1:nk, Weights(ws))     # Get random firm type that makes the offer pr. G | kk[i,t-1] and weighted by vacancy rate.
                        end
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
                    spellcount[i,t] = 0                     
                end

            end
            
        end

        #------------------------------------------------------------

        # This part likely to be the same:
        # we remove industries so jj == kk 
        jj = kk # Firm identifiers

        contiguous_ids = Dict( unique(jj) .=> 1:length(unique(jj))  )
        jj .= getindex.(Ref(contiguous_ids),jj);

        #----------------------------------------------------------------------------------------
        ii = repeat(1:ni,1,nt)
        tt = repeat((1:nt)',ni,1)
        df = DataFrame(i=ii[:], j=jj[:], l=ll[:], k=kk[:], α=α[ll[:]], ψ=ψ[kk[:]], v = v[kk[:]], t=tt[:], w_id=id[:] , spell=spellcount[:]);
        
        df[!, :lw] = df.α + df.ψ + w_sigma * rand(Normal(), size(df)[1]);
        df[!, :v_a] = df.v - df.ψ ;

        new(df, hyper_parameters, transition_matrix, kk)
    
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
    M = @chain link_df begin        # Use link_df, recall this is moves, so j =/= j_next
        groupby([:j, :j_next])      # Group by :j and next job
        combine(nrow => :count, [:j, :j_next] => ((x, y) -> ("M_" .* string.(y) .*"_" .* string.(x))) => :M)
    end
    return M
end


# %%
Random.seed!(1234);
sim_parameters = define_parameters(2.0, 1.0, 1.0, 0.1, 0.2, 0.2)
sim_hyper_parameters = define_hyper_parameters(4, 10, 10, 100_000, 0.8, 0.2, 1)
sim_transition_matrix = compute_transition_matrix(sim_parameters, sim_hyper_parameters)
# %%
sim_draw = simulation_draw(sim_hyper_parameters, sim_transition_matrix)
sim_df = sim_draw.df
df_connected = create_connected_df(sim_df)

# %% 
link_df     = find_firm_links(find_movers(sim_df))
M_df        = create_M_flows(link_df)
S_kk, M_0   = create_fixed_point_matrices(sim_df)

# %%
connected_df = lewd.create_connected_df(sim_df)
link_df = lewd.find_firm_links(lewd.find_movers(connected_df))
M_df = lewd.create_M_flows(link_df)
S_kk, M_0 = lewd.create_fixed_point_matrices(connected_df)
S_kk

M_df

# %%
rank = lewd.estimate_rank(S_kk, M_0)
# %%
v_rank = @chain connected_df begin
    groupby(:j)
    combine(:v => unique)
    sort(:v_unique)
end

rank_df = DataFrame([1:length(rank), log.(rank)], [:j, :rank])


comp_df = innerjoin(v_rank, rank_df, on = :j)
comp_df[!, "s_kk"] = diag(S_kk)
sort(comp_df, :s_kk)
# %%
cor(comp_df.v_unique, comp_df.rank)
corspearman(comp_df.v_unique, comp_df.rank)
# %%
cor(comp_df.v_unique, comp_df.s_kk)
corspearman(comp_df.v_unique, comp_df.s_kk)
comp_df


# %%

function estimate_v(df)
    connected_df = lewd.create_connected_df(df)
    S_kk, M_0 = lewd.create_fixed_point_matrices(connected_df)
    v_hat = lewd.estimate_rank(S_kk, M_0)
    return v_hat
end

function sim_many_times(N)
    comp_vec = Vector{DataFrame}(undef, N)
    for i in 1:N
        sim_i = simulation_draw(sim_hyper_parameters, sim_transition_matrix)
        sim_i_df = sim_i.df

        v_rank = @chain connected_df begin
            groupby(:j)
            combine(:v => unique)
            sort(:v_unique)
        end
        v_hat = estimate_v(sim_i_df)
        rank_df = DataFrame([1:length(v_hat), v_hat], [:j, :rank])


        comp_df = innerjoin(v_rank, rank_df, on = :j)
        comp_df[!, "draw_n"] .= i
        comp_vec[i] = comp_df
    end
    return comp_vec
end

many_sim_df = reduce(vcat, sim_many_times(100))


cor_df = @chain many_sim_df begin
    groupby(:draw_n)
    combine([:v_unique, :rank] => ((x, y) -> cor(x, y)) => :cor,
            [:v_unique, :rank] => ((x, y) -> corspearman(x, y)) => :rank_cor)
end
histogram(cor_df.rank_cor, bins = 10)

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

