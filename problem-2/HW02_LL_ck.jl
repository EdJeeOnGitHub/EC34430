################################################################################
# Labour Lamadon: Problem set 2
################################################################################

# Preparing environment ########################################################
# %%
using Pkg
Pkg.activate(".") # Create new environment in this folder 
# import packages 
using Distributions
using LinearAlgebra
using StatsBase
using DataFrames
using Plots
using CategoricalArrays
using ShiftedArrays
using Random
using Chain
using DataFramesMeta
using SparseArrays
using LightGraphs
# Create worker and firm types and initial covariance matrix ###################
# %%
struct simparams 
    nk::Int
    nl::Int
    nt::Int  
    ni::Int # number of workers at any given time period 
    fpt::Int
    λ::Float64
    function simparams(; nk=50, nl=10, nt=10, ni = 10_000, fpt = 4, λ = 0.5)
        new(nk, nl, nt, ni, fpt, λ) 
    end
end


draw_firm_from_type(k, firms_per_type) = sample(1:firms_per_type) + (k - 1) * firms_per_type
# function to create worker types
function f_wkreffect(;nwkr::Real, α_mean::Real=0, α_sd::Real=1)
    
    nl = nwkr
    α = quantile.(Normal(α_mean, α_sd), (1:nl) / (nl + 1))

    return α
end  #f_wkreffect

# function to draw ψ, V, f from joint normal distribution.
function f_firmchar(;nfirm::Real, ψ_mean::Real=0, v_mean::Real=0, f_mean::Real=0, 
    ψ_sd::Real=1, v_sd::Real=1, f_sd::Real=1, 
    ψv_corr::Real, ψf_corr::Real, vf_corr::Real)

    param_mean = [ψ_mean, v_mean, f_mean] 

    ψv_cov = ψv_corr * v_sd * ψ_sd
    ψf_cov = ψf_corr * ψ_sd * f_sd 
    vf_cov = vf_corr * v_sd * f_sd
    param_cov = [ψ_sd^2 ψv_cov ψf_cov 
                 ψv_cov v_sd^2 vf_cov 
                 ψf_cov vf_cov f_sd^2]
    
    # consider taking 10,000 draws and using quantiles instead
    firm_param = rand(MvNormal(param_mean, param_cov), nfirm)'

    ψ = firm_param[:,1]
    v = firm_param[:,2]
    f = firm_param[:,3]
    f = (exp.(f)) ./ sum(exp.(f))  # since f needs to be a distribution

    return ψ,v,f
end  # f_firmchar

# function for initial dist. of workers among firms 
function f_initdist(ψ, α; csort::Real=0.5, csig::Real=0.5,
    nwkr::Real, nfirm::Real)

    # initial cross-sectional sorting matrix
    # G[l,k]: prob. that worker of type l is in firm of type k
    G = zeros(nwkr, nfirm)
    for l in 1:nwkr 
        for k in 1:nfirm 
            G[l, k] = pdf( Normal(0, csig), ψ[k] - csort * α[l])
        end
        G[l,:] = G[l, :] ./ sum(G[l,:])  # G is a distribution 
    end

    return G 
end  # f_initdist
# function for moving decision
function f_simpanel(G,v,f, sim_params::simparams; δ::Real=0.02)
    nk, nl, nt, ni = sim_params.nk, sim_params.nl, sim_params.nt, sim_params.ni
    λ = sim_params.λ
    # We simulate a balanced panel
    ii = repeat(1:ni, 1, nt)  # Worker ID
    ll = zeros(Int64, ni, nt) # Worker type
    kk = zeros(Int64, ni, nt) # Firm type
    spellcount = zeros(Int64, ni, nt) # Employment spell
    vv = zeros(Float64, ni, nt)
    # iid logit shock for each individual in each time period 
    # each row is individual i's entire series of shocks
    ϵ = rand(Logistic(), ni, nt) 

    for i in 1:ni  # for each worker
        
        # set worker's type (randomly) for all time periods
        l = rand(1:nl)
        ll[i,:] .= l  # worker i's type is l 

        # time 1: draw from initial dist 
        kk[i,1] = sample(1:nk, Weights(G[l,:]))  # G[l,] is worker l's dist. over firms 
        vv[i, 1] = v[kk[i, 1]]
        # time 2 to nt 
        for t in 2:nt 
            # for now, no deaths
            if rand() < 0*δ  # if die, replace 
                l = rand(1:nl)  # draw type for new worker
                ll[i,t:nt] .= l  
                kk[i,t] = sample(1:nk, Weights(G[l,:]))  # draw firm type 
                ii[i,t] = ni + ii[i,t-1]  # new worker id for new worker (Rebecca)
                spellcount[i,t] = 0
                vv[i, t] = v[kk[i, t]]
            else           # if don't die, worker may face moving decision
                if rand() < λ  # have option to move
                    current = kk[i,t-1] 
                    offer = sample(1:nk, Weights(f))  
                    if v[current] < v[offer] + ϵ[i,t]  # if offer is better, move 
                        kk[i,t] = offer 
                        spellcount[i,t] = 1 + spellcount[i,t-1]
                        vv[i, t] = v[offer]
                    else                               # if current is better, stay
                        kk[i,t] = current
                        spellcount[i,t] = spellcount[i,t-1]
                        vv[i, t] = v[current]
                    end  # accept or reject offer
    
                else           # no option to move - stay at current firm
                    kk[i,t] = kk[i,t-1]
                    spellcount[i,t] = spellcount[i,t-1]
                    vv[i, t] = v[kk[i, t-1]]
                end  # receive offer or not
            end  # die or not 
        end  # loop thru each time period 
    end  # loop thru each worker 

    # make sure worker ids are contiguous 
    contiguous_ids = Dict( unique(ii) .=> 1:length(unique(ii))  )
    ii .= getindex.(Ref(contiguous_ids),ii);

    return ii,ll,kk,spellcount, vv
end  # f_simpanel
# plot 
# Plots.plot(G[:,:], xlabel="Firm type", ylabel="Worker type", 
#     zlabel="Probability", st=:wireframe)
# hmm, doesn't look right...
# function to assign firm ids 
function f_assign_firmid(ii,kk,ni,nt, nk, fpt, spellcount)

    # Array for firm id. Each entry is worker-year observation.
    jj = zeros(Int64, ni, nt) 
    
    for i in 1:ni  # for "each worker" (worker COULD die and be replaced)

        # period 1: extract firm type
        t = 1
        k = kk[i,t]  # even if kk not input, ok??? (in FL's fn)
        
        # we draw the firm id from the group of ids assigned to that firm type
        jj[i,t] = draw_firm_from_type(k, fpt)
 
        # for all subsequent periods
        for t in 2:nt
            # if employer and worker are the same
            if spellcount[i,t] == spellcount[i,t-1] && ii[i,t] == ii[i, t-1]
                jj[i,t] = jj[i,t-1]
            
            # if employer changed or worker is new
            else                                   
                k = kk[i,t]  # extract the (new) firm type 
                new_j = draw_firm_from_type(k, fpt)            
                # Make sure the new firm is actually new 
                while new_j == jj[i,t-1]  
                    new_j = draw_firm_from_type(k, fpt)
                    # ws = fill(1.0/(nk-1), nk)
                    # ws[k] = 0
                    # new_j = sample(1:nk, Weights(ws))
                end

                jj[i,t] = new_j
            end  # employer changed or not 
        end  # loop thru each time period
    end  # loop thru each worker 

    # Make sure firm ids are contiguous
    contiguous_ids = Dict( unique(jj) .=> 1:length(unique(jj))  )
    jj .= getindex.(Ref(contiguous_ids),jj);

    return jj
end  # f_assign_firmid

function f_gen_df(ni, nt, ll, jj,kk,α,ψ,spellcount, vv)
    ii = repeat(1:ni,1,nt)     # ni x nt
    tt = repeat((1:nt)',ni,1)  # ni x nt
    df = DataFrame(
        i=ii[:], 
        j=jj[:], 
        l=ll[:], 
        k=kk[:], 
        v = vv[:],
        α=α[ll[:]], 
        ψ=ψ[kk[:]], 
        t=tt[:], 
        spell=spellcount[:]);
    df[!, "w_id"] = df.i
    return df
end  # f_gen_df

global_seed = 1234
Random.seed!(global_seed)



function draw_sim(sim_params::simparams)
    nk, nl, nt, ni = sim_params.nk, sim_params.nl, sim_params.nt, sim_params.ni
    fpt = sim_params.fpt
    # number of types
    α = f_wkreffect(nwkr=nl)

    # Jt dist. of V and ψ should match Corr(V^EE,ψ)=0.400 in EE subsample. 
    # see table 2, column EE of Sorkin (2018)
    ψ,v,f = f_firmchar(nfirm=nk, ψv_corr=0.4, ψf_corr=0.1, vf_corr=0.05)


    # note that S := ψ[k] - csort * α[l] with csort > 0 generates positive sorting:
    # case ψ[k]>0, α[l]>0: then S probably closer to 0, so pdf(S) high 
    # case ψ[k]<0, α[l]<0: then S probably closer to 0, so pdf(S) high 
    # case ψ[k]>0, α[l]<0: then S probably >>0, so pdf(S) low
    # case ψ[k]<0, α[l]>0: then S probably <<0, so pdf(S) low

    G = f_initdist(ψ, α; nwkr=nl, nfirm=nk)


    ii,ll,kk,spellcount, vv = f_simpanel(G,v,f, sim_params)
    # note that since we replace dead workers, we have > 10,000 worker ids
    # but still 10,000 workers at any given time period


    # number of firms per type 

    # this fn randomly draws an id based on firm type

    jj = f_assign_firmid(ii,kk,ni,nt,nk, fpt, spellcount)


    df = f_gen_df(ni, nt,ll,  jj,kk,α,ψ,spellcount, vv)
    return df
end

# %%
include("./estimation-code.jl")
using .lewd

function estimate_v(df)
    connected_df = lewd.create_connected_df(df)
    S_kk, M_0 = lewd.create_fixed_point_matrices(connected_df)
    v_hat = lewd.estimate_rank(S_kk, M_0)
    return v_hat
end


function extract_true_v(df)
    v_rank = @chain df begin
        groupby(:j)
        combine(:v => unique)
    end
    size_df = @chain df begin
        groupby([:j, :w_id])
        combine(nrow => :count)
    end
    summ_df = innerjoin(
        v_rank,
        size_df,
        on = :j
    )
    return summ_df
end

################### Gen Data ##############
# %%
sim_params = simparams(nk = 4, ni = 50_000, fpt = 4)
df = draw_sim(sim_params)

# %%

# BUMP UP NK AS FRACTION


############## ED EDITS ###################

v_hat = estimate_v(df)
true_v_df = extract_true_v(df)


v_hat_df = DataFrame([1:length(v_hat), v_hat], [:j, :v_hat])


comp_df = innerjoin(true_v_df, v_hat_df, on = :j)
comp_df = @chain comp_df begin
    @transform(:old_v_hat = :v_hat, 
               :v_hat = :count .* :v_hat)
end


println("Spearman: $(corspearman(comp_df.v_unique, comp_df.v_hat))")
println("Corr: $(cor(comp_df.v_unique, comp_df.v_hat))")



######## Simulations #########
# %%
λ_grid = 0.2:0.1:0.95
nk_grid = 3:1:10
ni_grid = 10_000

function sim_over_grid(N, λ_grid, nk_grid, ni_grid)

    param_grid = collect(Iterators.product(λ_grid, nk_grid, ni_grid))[:]
    param_grid = repeat(param_grid, N)
    result_df = DataFrame(param_grid)
    rename!(result_df, [:λ, :nk, :ni])

    result_df[!, "cor"] .= 99.0
    result_df[!, "spearman_cor"] .= 99.0
Threads.@threads for i = 1:size(result_df, 1)
        param_realisation = param_grid[i]
        sim_param = simparams(
            λ = param_realisation[1],
            nk = param_realisation[2],
            ni = param_realisation[3],
            fpt = 2)
        sim_df = draw_sim(sim_param)

        v_hat = estimate_v(sim_df)
        true_v_df = extract_true_v(sim_df)


        v_hat_df = DataFrame([1:length(v_hat), v_hat], [:j, :v_hat])


        comp_df = innerjoin(true_v_df, v_hat_df, on = :j)
        
        comp_df = @chain comp_df begin
            @transform(:old_v_hat = :v_hat, 
                    :v_hat = :count .* :v_hat)
        end
        corr_spear = corspearman(comp_df.v_unique, comp_df.v_hat)
        corr = cor(comp_df.v_unique, comp_df.v_hat)
        result_df[i, "cor"] = corr
        result_df[i, "spearman_cor"] = corr_spear
    end
    return result_df
end



# %%
function sim_metrics(df, col; create_plot = true)
    mean_df = @chain df begin
        groupby(col)
        @combine(
            :mean_cor = mean(:cor),
            :mean_cor_rank = mean(:spearman_cor),
            :sd_cor = var(:spearman_cor)
        )
    end

    if create_plot == true
        p = plot(
            mean_df[!, col],
            mean_df.mean_cor_rank,
            linetype = :scatter,
            label = "",
            title = "Spearman Correlation, V and V̂, varying $col",
            ylabel = "Spearman Correlation",
            xlabel = "$col"
            # yerror = sqrt.(mean_df.sd_cor)*1.96
        )
        return mean_df, p
    end

    return mean_df
end
# %%
nk_sims = sim_over_grid(1_000, 0.25, 3:1:10, 10_000)
# %%
n_sims = sim_over_grid(1, 0.25, 20, 10_000:10_000:50_000)
# %%
λ_sims = sim_over_grid(500, λ_grid, 10, 10_000)
# %%
nk_metrics, nk_plot = sim_metrics(nk_sims, :nk)
n_metrics, n_plot = sim_metrics(n_sims, :ni)
λ_metrics, λ_plot = sim_metrics(λ_sims, :λ)
# %%
nk_plot
n_plot
λ_plot
# %%
n_sims


# %%
function plot_sim_results(df)
    best_fi
    
end


mean_df = @chain λ_sims begin
    groupby(:λ)
    @combine(
        :mean_cor = mean(:cor),
        :mean_rank_cor = mean(:spearman_cor)
     )
end

plot(
    mean_df.λ,
    mean_df.mean_cor,
    linetype = :scatter
)
# %%
############### CLARAS STUFF NOT ED ANYMORE #################


#############################
# extract connected component (this part thanks to Rebecca Wu)
# adj_df = @chain df begin
#     groupby([:i]) 
#     @transform(:j_lead = lead(:j))
#     transform([:j_lead, :j] => ByRow(coalesce) => :j_lead)
#     @transform(:move_dum = (:j_lead .!= :j) .* 1) 
#     transform(:move_dum => (x -> ifelse.(ismissing.(x), 0, x)) => :move_dum)
#     # keep all movers 
#     @subset(:move_dum .== 1) 
#     sparse(_.j, _.j_lead, ones(size(_)[1]))
# end 

# adj_matrix = ((adj_df + adj_df') .> 0) .* 1
# connected_components(SimpleGraph(adj_matrix))[1] # all firms in connected set 
# ###########################

# # do we need to get firm size?


# ################################################################################
# # if what we have to do is eq'n 6

# # function to get flow matrix M and exits matrix S (from row to col)
# function f_flow_matrix(df)
#     nfirms = length(unique(df[!,:j]))

#     M = zeros(Int64, nfirms, nfirms)
#     for k in 1:nfirms, j in 1:nfirms  # moves from k to j
#         df_flow = @chain df begin
#             @subset((:j_l1 .== k) .& (:j .== j))
#         end
#         M[k,j] = nrow(df_flow)
#     end

#     S = zeros(Int64, nfirms, nfirms)
#     for k in 1:nfirms  # exits from k
#         df_exits = @chain df begin
#             @subset((:j_l1 .== k) .& (:j .!= k))
#         end
#         S[k,k] = nrow(df_exits)
#     end

#     return M,S
# end

# M,S = f_flow_matrix(df)

# # fixed point iteration (eq'n 6) to get estimate of tildeV
# # trying 20 iterations for now bc it's diverging
# function f_est_value(M,S) 

#     tol = 10 
#     n_iter = 0
#     nfirms = size(M,1)
#     tildeV_old = ones(Float64,nfirms,1)
#     tildeV_new = ones(Float64,nfirms,1)

#     # graph convergence (or lack thereof?) - just for firm 50
#     conv = ones(Float64,20,1)

#     while n_iter < 20
#         tildeV_new = inv(S) * M * tildeV_old
#         tol = maximum((tildeV_new - tildeV_old).^2)
#         tildeV_old = copy(tildeV_new)

#         n_iter = n_iter + 1

#         # so i can graph convergence 
#         conv[n_iter] = tildeV_new[50]
#     end
#     print("Number of iterations: $n_iter. Tolerance: $tol")

#     # return conv array too so i can graph 
#     return tildeV_new,conv
# end

# tildeV_new,conv = f_est_value(M,S)
# plot(1:20,conv)

# # value is diverging to infinity RIP

# # tildeV_new = f_est_value(M,S)

# ################################################################################
