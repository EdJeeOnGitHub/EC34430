
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

function sigmoid(z)
    return 1.0 ./ (1.0 + exp(-z))
end


function wls_lik(β, y, w_ijk, K, N)

    μ = β[1]
    σ = sigmoid.(β[2])
    lik_i = pdf.(Normal(μ,σ), y)
    logL = sum(log.(lik_i.*w_ijk))/(N)

    return -logL

end


function generate_true_parameters(K, T)  

    μ  = rand(Uniform(), (K, T)).*6
    σ  = ones((K,T))
    pk = rand(Dirichlet(ones(K)))
    
    return μ, σ, pk

end

function generate_fake_data(μ, σ, pk, N, T, K)

    # Y placeholder:
    Y = zeros(N, T) 

    for ii in 1:N
        # draw K
        k = sample(1:K, Weights(pk))
        for tt in 1:T
            # draw Y1, Y2, Y3
            Y[ii,tt]  = rand(Normal(μ[k,tt] ,σ[k,tt]))
        end
    end

    return Y

end


N =10000
T = 1
K = 3
nIter = 10000
ω_ijk = zeros(N,K)
lpm = zeros(N,K)

# Model parameters
# Type probabilities

# μ_true, σ_true, pk = generate_true_parameters(K, T);
μ_true = [0.2, 4, 8]
σ_true = [1,1, 1].*.5
pk = [0.2,0.4,0.4]
Y = generate_fake_data(μ_true, σ_true, pk, N, T, K)

histogram(Y, bins=100)

μ_kt = rand(K,T)
σ_kt = rand(K,T)
p_k = rand(Dirichlet(ones(K)))

ω_num_k = zeros(N, K)
logLikList = zeros(nIter)
logLikList[1] = 4

tol = 0.0001

for rr in 3:nIter

    # Compare lik_Lag vs Lik now,
    # if less than tol, end.
    if abs(logLikList[rr - 2] - logLikList[rr-1]) > tol 

        println(abs(logLikList[rr - 2] - logLikList[rr-1]))

        lik = 0
        for ii in 1:N
            for kk in 1:K
                # Get pdf evaluated at each period cond on μ_kt and σ_kt:
                norm_pdf_k = pdf.(Normal.(μ_kt[kk,:], σ_kt[kk,:]) , Y[ii,:])
                ω_num_k[ii,kk] = p_k[kk]*prod(norm_pdf_k) 
            end
            ω_ijk[ii,:] = ω_num_k[ii,:]./sum(ω_num_k[ii,:])
            # lik += log(sum(ω_num_k[ii,:]))
        end

        mixture = MixtureModel(Normal, [(μ_kt[k], σ_kt[k]) for k in 1:K], p_k)
        lik += sum(logpdf(mixture, Y))
        logLikList[rr] = lik

        # The M - maximization step:
        # Identity = Matrix(I, K, K)
        p_k = mean(ω_ijk, dims=1)[:] #I believe this part is wrong

        for tt in 1:T

            # Use Wiemann's method:
            # μ_hat = sum(Y[:,tt] .* ω_ijk, dims=1)./sum(ω_ijk, dims=1)
            # σ_hat = sum((Y[:,tt] .- μ_kt[:,tt]').^2 .*ω_ijk, dims=1)./sum(ω_ijk, dims=1)
            # μ_kt[:,tt] = μ_hat
            # σ_kt[:,tt] .= 1
            # σ_kt[:,tt] = sqrt.(σ_hat)

            # Mu and sigma:
            for kk in K
                β=zeros(2)
                β[1]=μ_kt[kk,tt]
                β[2]=σ_kt[kk,tt]
                func_μ(β) = wls_lik(β, Y, ω_ijk[:,kk], K, N)
                res = optimize(func_μ, β, LBFGS(), Optim.Options(iterations = 1000))
                β_hat = Optim.minimizer(res)

                μ_kt[kk,tt] = β_hat[1]
                σ_kt[kk,tt] = sigmoid.(β_hat[2])
            end
        
        end
    else
        break
    end

end


sigmoid.(μ_hat[K+1:end])

μ_true

μ_kt

σ_true

σ_kt

scatter(sort(μ_true[:]), sort(μ_kt[:]))
scatter(sort(p_k[:]), sort(pk[:]))

# Generate fake data:



histogram(Y, bins=100)