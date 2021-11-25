
using Pkg
Pkg.activate(".") # Create new environment in this folder

using Distributions
using LinearAlgebra
using StatsBase
using DataFrames
using Plots
using Optim
using Random
using SparseRegression


function wls_obj(β, y, x, w_ijk)
    return sum(w_ijk.*(y .- x*β).^2) 
end


N=1000
T=4
K = 3
ω_ijk = zeros(N,K)
lpm = zeros(N,K)

# Model parameters
# Type probabilities
p_k = rand(Dirichlet(ones(K)))
μ_kt = rand(K,T)
σ_kt = rand(K,T)

Y = rand(N, T)

ω_num_k = zeros(N, K)

for kk = 1:nIter

    lik = 0
    for ii in 1:N
        for kk in 1:K
            # Get pdf evaluated at each period cond on μ_kt and σ_kt:
            norm_pdf_k = pdf.(Normal.(μ_kt[kk,:], σ_kt[kk,:]) , Y[ii,:])
            ω_num_k[ii,kk] = p_k[kk]*prod(norm_pdf_k) # p_k × ∏_t ϕ(y_{ii,t})
        end
        lik  += sum(log.(ω_num_k[ii,:]))
        ω_ijk[ii,:] = ω_num_k[ii,:]./sum(ω_num_k[ii,:])
    end

    # The M - maximization step:
    Identity = Matrix(I, K, K)
    for tt in 1:T

        DY     = kron(Y[:,1], ones(K))
        Dkj    = kron(ones(N),Identity)

        # Mu:
        func(β) = wls_obj(β, DY, Dkj, w_ijk)
        params0 = rand(K)
        res = optimize(func, params0, LBFGS(), Optim.Options(iterations = 1000))
        β_hat = Optim.minimizer(res)

        # Get residual
        resid = ((DY - Dkj*params_hat).^2)./w_ijk
        # Sigma:
        func(σ) = wls_obj(σ, resid, Dkj1, w_ijk)
        params0 = rand(K)
        res = optimize(func, params0, LBFGS(), Optim.Options(iterations = 1000))
        σ_hat = Optim.minimizer(res)


        #Update:
        μ_kt[:,tt] = sqrt.(params_hat)
        σ_kt[:,tt] = sqrt.(σ_hat)
    end

    # Compare lik_Lag vs Lik now,
    # if less than tol, end.
end


wls_obj(params0, DY1, Dkj1, w_ijk)

sum(log.(w_ijk.*pdf(Normal(0,1),(DY1 .- Dkj1*β0)./σ0)))