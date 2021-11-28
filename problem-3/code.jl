using Random
using StatsBase
using Distributions
using LinearAlgebra
function lognormpdf(Y; mu = 0,sigma = 1)
    l = -0.5*((Y .- mu)/sigma).^2 .- 0.5*log(2*pi) .- log.(sigma)
    return l
end

function logsumexp(v)
    vm = maximum(v)
    lse = log(sum(exp.(v .- vm))) + vm
    return lse 
end



function hugo_lognormpdf(Y,#::Array{Float64},
                    mu,#::Array{Float64},
                    sigma)#::Array{Float64})

    -0.5 * (  (Y-mu) / sigma )^2   - 0.5 * log(2.0*pi) - log(sigma)  

end
#end   

struct hugo_logsumexp
    #v::Array{Float64}

    function hugo_logsumexp(v)#::Array{Float64})

    vm = max(v)
    log(sum(exp(v-vm))) + vm
    end
end

function sim_data(k,n, T; doplot = false)
    # true values
    μ = randn(k)*10
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


ed_data = sdata(5, 1000, 5)
ed_data[:y][:, 1]
using Plots

ed_data[:μ]
histogram(ed_data[:y][:, 1], bins = 50)

function expectation(Y, p_k, μ_k, σ_k)
    # Y = rand(10, 3)
    # p_k = [1/3, 1/3, 1/3]
    # μ_k = [1, 1, 1]
    # σ_k = [1, 2, 3]
    N = size(Y, 1)
    T = size(Y, 2)
    K = size(p_k, 2)
    
    LL = Matrix{Float64}(undef, (N, K))
    posterior = Matrix{Float64}(undef, (N, K))
    for k = 1:K 
        LL[:, k] = sum(log.(p_k[:, k]) .+ lognormpdf(Y, mu = μ_k[k], sigma = σ_k[k]), dims = 2)
    end
    posterior[:,:] = exp.(LL .- logsumexp(LL))
    # posterior[:,:] = exp.(LL) ./ sum(exp.(LL), dims = 2)
    return posterior
end


## Hugo code

struct fake_hugo_data
    Y1::Vector
    Y2::Vector
    Y3::Vector
    μ0::Array
    π0::Array
    σ0::Array
    μ::Array
    π::Array
    σ::Array
    function fake_hugo_data()
    N = 10
    nk = 5
    tau = Array{Float64, 2}(undef, N, nk)
    lpm = Array{Float64, 2}(undef, N, nk)
    # Now I'm going to randomly generate my Y's from a mixture of Gaussians with a
    π0 = fill(1/(nk), ((nk),1))
    # This gives the initial values for the std dev of each Gaussian model
    σ0 = repeat(fill(1, ((nk),1)), 3)
        σ0 = reshape(σ0,3,(nk))
    # This gives the initial values for the means of each Gaussian model
    μ0 = repeat(collect(1:(nk)),inner = 3)
    μ0 = reshape(μ0, 3,(nk))
    # Simulating the Sample from the true DGP constructed from π0, μ0 and σ0
    Y = rand(MvNormal(μ0[1,:], I), 3*N)
        Y = π0' * Y
        Y = Y'

    Y1 = Y[1:N]
    Y2 = Y[N+1:2*N]
    Y3 = Y[2*N+1:3*N]

    # This denotes the initial probability assigned to each Gaussian model
    π = fill(1/nk, (nk,1))
    # This gives the initial values for the std dev of each Gaussian model
    σ = Array{Float64}(repeat(fill(1, (nk,1)),3))
        σ = reshape(σ,3,nk)
    # This gives the initial values for the means of each Gaussian model
    μ = Array{Float64}(repeat(collect(1:nk),inner = 3))
        μ = reshape(μ, 3,nk)
        new(Y1, Y2, Y3, μ0, π0, σ0, μ, π, σ )
    end
end



function Expectation(Y1,
                     Y2,
                     Y3,
                     μ,
                     σ,
                     τ,
                     N,
                     nk)

                     lnorm1 = Array{Float64, 2}(undef, N, nk)
                     lnorm2 = Array{Float64, 2}(undef, N, nk)
                     lnorm3 = Array{Float64, 2}(undef, N, nk)
                     lall = Array{Float64, 2}(undef, N, nk)
                     lik = Array{Float64, 2}(undef, N, nk)
                     lpm = Array{Float64, 2}(undef, N, nk)


    lτ = log.(τ)
    for i in 1:N  
        lnorm1[i,:] = hugo_lognormpdf.(Y1[i], μ[1,:], σ[1,:])
        lnorm2[i,:] = hugo_lognormpdf.(Y2[i],μ[1,:],σ[1,:])
        lnorm3[i,:] = hugo_lognormpdf.(Y3[i],μ[1,:],σ[1,:])
    end
    lall = lτ + lnorm1 + lnorm2 + lnorm3
    lik = lik + hugo_logsumexp.(lall)

    for i in 1:N  
        τ[i,:] = exp.(lall[i,:]) ./ sum(exp.(lall[i,:]))
    end

    return(τ,lik)
    
end







# TODO: This step is probably wrong?
function maximisation(Y, posterior)
    N = size(Y, 1)
    p_k = vec(sum(posterior, dims = 1) ./ N)
    μ = vec(sum(posterior .* mean(Y,dims = 2), dims = 1) ./ sum(posterior, dims = 1))
    σ = vec(sqrt.(sum(posterior .* (mean(Y, dims = 2) .- μ').^2, dims = 1 ) ./ sum(posterior, dims = 1)))
    return p_k, μ, σ
end

Y_test = randn(10, 3)
p_test = fill(1/3, 10, 3)
μ_test = [1, 2, 3]
σ_test = [1, 2, 3]

e_1 = expectation(Y_test, p_test,μ_test, σ_test)
p_1 = maximisation(Y_test, e_1)
p_1

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
        old_p
        μ
        σ
    end
    return Dict(:p_k => p, :μ => μ, :σ => σ)
end


fake_data = sdata(3, 100, 3)
expectation_maximisation(fake_data[:y], 3)
fake_data






fake_hugo_df = fake_hugo_data()


fieldnames(fake_hugo_data)
hugo_τ = reshape(repeat(fake_hugo_df.π, 1000), 1000, 5)
hugo_post, hugo_lik  = Expectation(
    fake_hugo_df.Y1,
    fake_hugo_df.Y2,
    fake_hugo_df.Y3,
    fake_hugo_df.μ,
    fake_hugo_df.σ,
    hugo_τ,
    1000,
    5 
    )



ed_Y = hcat(fake_hugo_df.Y1, fake_hugo_df.Y2, fake_hugo_df.Y3)

ed_post = expectation(ed_Y, hugo_τ, fake_hugo_df.μ[1, :], fake_hugo_df.σ[1, :])


hugo_post
ed_post



