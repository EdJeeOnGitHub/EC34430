using Random
using StatsBase
using Distributions
using LinearAlgebra
function lognormpdf(Y; mu = 0,sigma = 1)
    l = -0.5*((Y .- mu)/sigma).^2 .- 0.5*log(2*pi) .- log(sigma)
    return l
end

function logsumexp(v)
    vm = maximum(v)
    lse = log(sum(exp.(v .- vm))) + vm
    return lse 
end

function sdata(k,n, T; doplot = false)
    Random.seed!(3333)
    # true values
    μ = randn(k)
    σ = abs.(randn(k))
    p = rand(k)
    p = p / sum(p)
    K_draw = sample(1:k, Weights(p), n, replace = true)
    Y = Matrix{Float64}(undef, n, T)
    for t in 1:T
        Y[:, t] = rand.(Normal.(μ[K_draw], σ[K_draw])) 
    end

    return Dict(:y => Y, :μ => μ, :σ => σ, :p => p)
end





function expectation(Y, p_k, μ_k, σ_k)
    N = size(Y, 1)
    T = size(Y, 2)
    K = size(p_k, 1)
    if T < K 
        error("Probably not identified here.")
    end

    LL = Matrix{Float64}(undef, (N, K))
    posterior = Matrix{Float64}(undef, (N, K))
    for k = 1:K 
        LL[:, k] = sum(log(p_k[k]) .+ lognormpdf(Y, mu = μ_k[k], sigma = σ_k[k]), dims = 2)
    end

    posterior[:,:] = exp.(LL .- logsumexp(LL))
    return posterior
end

# TODO: This step is probably wrong?
function maximisation(Y, posterior)
    Y = Y_test 
    posterior = e_1  
    N = size(Y, 1)
    p_k = vec(sum(posterior, dims = 1) ./ N)
    μ = vec(sum(posterior' .* Y, dims = 1) ./ sum(posterior, dims = 1))
    σ = vec(sqrt.(sum(posterior' .* (Y .- μ').^2, dims = 1 ) ./ sum(posterior, dims = 1)))
    return p_k, μ, σ
end

Y_test = randn(10, 3)
p_test = [1/3, 1/3, 1/3]
μ_test = [1, 2, 3]
σ_test = [1, 2, 3]

e_1 = expectation(Y_test, p_test,μ_test, σ_test)
p_1 = maximisation(Y_test, e_1)


function expectation_maximisation(Y, K)
    Y = fake_data[:y]
    K = 3
    μ = rand(K)
    p = rand(K)
    p = p ./ sum(p) 
    σ = abs.(rand(K))
    diff = Inf
    iter = 0
    while diff > 1e-6
        iter += 1
        println("iter: $iter")
        old_p, old_μ, old_σ = p, μ, σ
        e_step = expectation(Y, p, μ, σ )
        p, μ, σ = maximisation(Y, e_step)
        p_diff = maximum(abs.(p - old_p))
        μ_diff = maximum(abs.(μ - old_μ))
        σ_diff = maximum(abs.(σ - old_σ))
        diff = maximum([p_diff,μ_diff, σ_diff])
        p 
        μ
        σ
    end
    return Dict(:p_k => p, :μ => μ, :σ => σ)
end


fake_data = sdata(3, 10, 3)
fake_data[:y]


expectation_maximisation(fake_data[:y], 3)
