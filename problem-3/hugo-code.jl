# %%
using Random, 
      Distributions,
      LinearAlgebra,
      StatsBase
Random.seed!(123)
function lognormpdf(Y,#::Array{Float64},
                    mu,#::Array{Float64},
                    sigma)#::Array{Float64})

    -0.5 * (  (Y-mu) / sigma )^2   - 0.5 * log(2.0*pi) - log(sigma)  

end

function logsumexp(v)#::Array{Float64})
    vm = max(v)
    log(sum(exp(v-vm))) + vm
end


function WLS(X,
                Y,
                Ω)

    # Note that this takes in the variance covariance matrix in vector form
    # Note that Omega must be a diagonal matrix for this to work
    invΩ = 1 ./ Ω
    invΩ = diagm(invΩ)

    β = inv(X'*invΩ*X)*(X'*invΩ*Y)
            
    return β
end

function Expectation(Y1,
                     Y2,
                     Y3,
                     μ_input,
                     σ_input,
                     τ_input,
                     N,
                     nk)
    lnorm1 = Array{Float64, 2}(undef, N, nk)
    lnorm2 = Array{Float64, 2}(undef, N, nk)
    lnorm3 = Array{Float64, 2}(undef, N, nk)
    lall = Array{Float64, 2}(undef, N, nk)
    lik = Array{Float64, 2}(undef, N, nk)

    lτ = log.(τ_input)
    τ_output = similar(τ_input)
    for i in 1:N  
        lnorm1[i,:] = lognormpdf.(Y1[i],μ_input[1,:],σ_input[1,:])
        lnorm2[i,:] = lognormpdf.(Y2[i],μ_input[2,:],σ_input[2,:])
        lnorm3[i,:] = lognormpdf.(Y3[i],μ_input[3,:],σ_input[3,:])
    end
    lall = lτ + lnorm1 + lnorm2 + lnorm3
    lik = lik + logsumexp.(lall)
    
    for i in 1:N  
        τ_output[i,:] = exp.(lall[i,:]) ./ sum(exp.(lall[i,:]))
    end
    
    return τ_output, lik
end

function Maximization(Y1, 
                Y2,
                Y3,
                τ, 
                N,
                nk)

μ_output = Matrix{Float64}(undef, (3,nk))
σ_output = Matrix{Float64}(undef, (3,nk))
    
DY1 = kron(Y1,fill(1, (nk,1)))
DY2 = kron(Y2,fill(1, (nk,1)))
DY3 = kron(Y3,fill(1, (nk,1)))

Dkj1 = kron(fill(1, (N,1)), 1* Matrix(I, nk, nk))
Dkj2 = kron(fill(1, (N,1)), 1* Matrix(I, nk, nk))  
Dkj3 = kron(fill(1, (N,1)), 1* Matrix(I, nk, nk))

Ω1 = vec(reshape(τ', size(DY1, 1),1))
Ω2 = vec(reshape(τ', size(DY2, 1),1))
Ω3 = vec(reshape(τ', size(DY3, 1),1))



WLS1 = WLS(Dkj1, DY1, Ω1)
WLS2 = WLS(Dkj2, DY2, Ω2)
WLS3 = WLS(Dkj3, DY3, Ω3)

for i in 1:nk  
μ_output[1,i] = 1/N*sum(Dkj1[:,1].*WLS1[i])
σ_output[1,i] = sqrt(1/N*sum((Y1 - Dkj1[:,1][Dkj1[:,1].>0].*WLS1[i]).^2))
end
for i in 1:nk  
μ_output[2,i] = 1/N*sum(Dkj2[:,1].*WLS2[i])
σ_output[2,i] = sqrt(1/N*sum((Y2 - Dkj2[:,1][Dkj2[:,1].>0].*WLS2[i]).^2))
end
for i in 1:nk  
μ_output[3,i] = 1/N*sum(Dkj3[:,1].*WLS3[i])
σ_output[3,i] = sqrt(1/N*sum((Y3 - Dkj3[:,1][Dkj3[:,1].>0].*WLS3[i]).^2))
end

return μ_output, σ_output
end

# %% DGP


function sim_data(k,n, T; doplot = false)
    # true values
    μ = randn(k)*10 .+ collect(1:k)
    σ = abs.(randn(k))
    p = rand(k)
    p = p / sum(p)
    K_draw = sample(1:k, Weights(p), n, replace = true)
    Y = Matrix{Float64}(undef, n, T)
    for t in 1:T
        Y[:, t] = rand.(Normal.(μ[K_draw], σ[K_draw])) 
    end

    return Dict(:y => Y, :μ => μ, :σ => σ, :p => p, :K_draw => K_draw)
end

function create_initial_values(k, n, T)
    initial_μ = Array{Float64}(repeat(collect(1:k),inner = T))
    initial_μ = reshape(initial_μ, T,k)

    initial_σ = Array{Float64}(repeat(fill(1, (k,1)),T))
    initial_σ = reshape(initial_σ,T,k)

    initial_τ = fill(1/k, (k, 1))
    initial_τ = repeat(initial_τ, n)
    initial_τ = reshape(initial_τ, n, k)
    return initial_μ, initial_τ, initial_σ
end


# EM funcs

function em!(Y1, Y2, Y3, μ, σ, τ)
    N = length(Y1)
    nk = size(μ, 2)
    exp_update = Expectation(Y1, Y2, Y3, μ, σ, τ, N, nk)
    τ_update = exp_update[1]
    # lik_update = sum(exp.(exp_update[2]))

    max_update = Maximization(Y1, Y2, Y3, τ_update, N, nk)

    μ_update = max_update[1]
    σ_update = max_update[2]
    return μ_update, σ_update, τ_update
end

fake_sim_data = sim_data(5, 500, 3)
Y_test = fake_sim_data[:y]
Y1_test = Y_test[:, 1]
Y2_test = Y_test[:, 2]
Y3_test = Y_test[:, 3]
μ_test, τ_test, σ_test = create_initial_values(5, 500, 3)
μ_output, σ_output, τ_output = em!(Y1_test, Y2_test, Y3_test, μ_test, σ_test, τ_test)

println("μ_output: $μ_output")
println("μ_test: $μ_test")

function find_diff(x, new_x)
    maximum(abs.(x .- new_x))
end

function em(Y, k)
    T = size(Y, 2)
    if T != 3
        error("You've hardcoded T = 3 dumbass")
    end
    n = size(Y, 1)
    μ, τ, σ = create_initial_values(k, n, T)
    diff = Inf
    iter = 0
    while diff > 1e-6
        iter += 1
        println("Iter: $iter")
        new_μ, new_σ, new_τ = em!(Y[:, 1], Y[:, 2], Y[:, 3], μ, σ, τ)
        μ_diff = find_diff(μ, new_μ)
        σ_diff = find_diff(σ, new_σ)
        τ_diff = find_diff(τ, new_τ)
        diff = maximum([μ_diff, σ_diff, τ_diff])
        if (iter > 200)
            return new_μ, new_σ, new_τ
        end
    end
    return new_μ, new_σ, new_τ
end

res_μ, res_σ, res_τ = em(Y_test, 5)
res_μ
fake_sim_data[:μ]

_, max_inds = findmax(res_τ, dims = 2)
most_likely_k = [ind[2] for ind in max_inds]
most_likely_k


[sum(fake_sim_data[:K_draw] .== k) for k in 1:3]
[sum(most_likely_k[:] .== k) for k in 1:3]
fake_sim_data[:p]

using Plots
histogram(Y_test[:,3], bins = 60)
res_τ == maximum

em!(Y_test[:, 1], Y_test[:, 2], Y_test[:, 3], μ_test, σ_test, τ_test)