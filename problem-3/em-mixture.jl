
using Pkg
Pkg.activate(".") # Create new environment in this folder

using Distributions
using LinearAlgebra
using StatsBase
using DataFrames
using Plots
using Optim
using Random


function wls_obj(β, y, x, w_ijk)
    return sum(w_ijk.*(y .- x*β).^2) 
end


N =1000
T = 4
K = 3
nIter = 1000
ω_ijk = zeros(N,K)
lpm = zeros(N,K)

# Model parameters
# Type probabilities
p_k = rand(Dirichlet(ones(K)))
μ_kt = rand(K,T)
σ_kt = rand(K,T)

Y = rand(N, T)

ω_num_k = zeros(N, K)
logLikList = zeros(nIter)
logLikList[1] = 4

tol = 0.0001

for kk in 3:nIter

    # Compare lik_Lag vs Lik now,
    # if less than tol, end.
    if abs(logLikList[kk - 2] - logLikList[kk-1]) > tol 

        println(logLikList[kk - 1] - logLikList[kk])

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
        logLikList[kk] = lik

        # The M - maximization step:
        Identity = Matrix(I, K, K)
        for tt in 1:T

            DY     = kron(Y[:,tt], ones(K))
            Dkj    = kron(ones(N),Identity)

            # Mu:
            w = ω_ijk[:]
            func_μ(β) = wls_obj(β, DY, Dkj, w)
            params0 = rand(K)
            res = optimize(func_μ, params0, LBFGS(), Optim.Options(iterations = 1000))
            μ_hat = Optim.minimizer(res)

            # Get residual
            resid = ((DY - Dkj*μ_hat).^2)./w

            # Sigma:
            func_σ(σ) = wls_obj(σ, resid, Dkj, w)
            params0 = rand(K)
            res = optimize(func_σ, params0, LBFGS(), Optim.Options(iterations = 1000))
            σ_hat = Optim.minimizer(res)

            #Update:
            μ_kt[:,tt] = μ_hat
            σ_kt[:,tt] = sqrt.(σ_hat)
        end
    else
        break
    end

end


