# PSET 3: Estimating Mixture Models #
# Danielle Nemschoff #

# Note: code adapted from Thibaut's R code version and Florian Oswald #

# ------------------------------------------------------------------------------------------------------------- #
# Set Up ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------- #
# models of wage formaation with unobserved heterogeneity in firms and workers #

# firms have types l: 1,...,L
# workers have types by k: 1,...,K

# worker i is of type k and workers for firm l in a certain period 
# wages are drawn from the normal distribution N(μkl, σkl)

# one firm type and 2 worker types 

# add time dimension 

using(RCall)
using(Distributions)
using(StatsFuns)
using(Random)
using(Plots)
using(DataFrames)

# ------------------------------------------------------------------------------------------------------------- #
# Data Generator Function ------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------- #
# sets true mixture params 
# creates a distribution mixture mmodoel
# draws n random realizations 


function sdata(n; t=40, doplot = false)
    Random.seed!(6)

    # true values
    μ = [2.0,5.0]
    σ = [0.5,0.7]
    pk = [0.3,0.7]

    m = MixtureModel([Normal(μ[i], σ[i]) for i in 1:2], pk)
    if doplot
        plot(
            plot(m,linewidth=2), 
            plot(m,linewidth=2, fill=(0,:red,0.5), components = false, title="Mixture"),dpi = 300
            )
        #savefig("mixtures.png")
    end
    y = rand(m,n)

    return Dict(:y => y, :μ => μ, :σ => σ, :pk => pk)
end

# ------------------------------------------------------------------------------------------------------------- #
# Estimation Function ----------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------- #
# 1. Take vector y
# 2. set (the same) wrong starting values
# 3. run the EM algorithm for iters iterations to find true values of 
#    proportion weights pk, means \mu and variances \sigma for each component.

function bm_jl(y::Vector{Float64}; t= 40, iters=100)
    y = fake_data[:y]
    # poor starting values
    μ = [4.0,6.0]
    σ = [1.0,1.0]
    pk = [0.5,0.5]

    N = length(y)
    K = length(μ)

    # initialize objects    
    L = zeros(N,K)
    p = similar(L)

    for it in 1:iters

        dists = [Normal(μ[ik], σ[ik] ) for ik in 1:K]

        # evaluate likelihood for each type 
        for i in 1:N
            for k in 1:K
                L[i,k] = log(pk[k]) + logpdf.(dists[k], y[i]) 
            end
        end
        L
        # get posterior of each type 
        p[:,:] = exp.(L .- logsumexp(L))
      
        # with p in hand, update 
        pk[:] .= vec(sum(p,dims=1) ./ N)
        μ[:] .= vec(sum(p .* y, dims = 1) ./ sum(p, dims = 1))
        σ[:] .= vec(sqrt.(sum(p .* (y .- μ').^2, dims = 1) ./ sum(p, dims = 1)))
        pk
    end
    return Dict(:pk => pk, :μ => μ, :σ => σ)
end

# ------------------------------------------------------------------------------------------------------------- #
# Simulate ---------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------- #

fake_data = sdata(10, t=40)

bm_jl(fake_data[:y], t = 40, iters=100)
using GaussianMixtures
function bm_jl_GMM(y::Vector{Float64};iters=100)
    gmm = GMM(2,1)  # initialize an empty GMM object
    # stick in our starting values
    gmm.μ[:,1] .= [4.0;6.0]
    gmm.Σ[:,1] .= [1.0;1.0]
    gmm.w[:,1] .= [0.5;0.5]

    # run em!
    em!(gmm,y[:,:],nIter = iters)
    return gmm
end

res = bm_jl_GMM(fake_data[:y])
res
res.μ
res.w


# ------------------------------------------------------------------------------------------------------------- #
# Implement on PSID ------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------- #

#data = data.table(read.dta13("~/Dropbox/Documents/Teaching/ECON-24030/lectures-laborsupply/homeworks/data/AER_2012_1549_data/output/data4estimation.dta"))
