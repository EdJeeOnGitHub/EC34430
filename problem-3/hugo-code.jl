# %%
using Random, 
      Distributions,
      LinearAlgebra
      
#struct lognormpdf 
    #Y::Array{Float64}
    #mu::Array{Float64}
    #sigma::Array{Float64}

    function lognormpdf(Y,#::Array{Float64},
                        mu,#::Array{Float64},
                        sigma)#::Array{Float64})

        -0.5 * (  (Y-mu) / sigma )^2   - 0.5 * log(2.0*pi) - log(sigma)  

    end
#end   

struct logsumexp
    #v::Array{Float64}

    function logsumexp(v)#::Array{Float64})

    vm = max(v)
    log(sum(exp(v-vm))) + vm
    end
end

struct reciprocal 
    function reciprocal(x)
        recip = 1/x
    end
end

struct WLS

    function WLS(X,
                 Y,
                 Ω)
    
    # Note that this takes in the variance covariance matrix in vector form
    # Note that Omega must be a diagonal matrix for this to work
    invΩ = reciprocal.(Ω)
        invΩ = diagm(invΩ)

    β = inv(X'*invΩ*X)*X'*invΩ*Y
            
    return(β)
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
    
        Y1 = Y1_test
        Y2 = Y2_test
        Y3 = Y3_test 
        μ = μ_test
        σ = σ_test
        π = π_test
        N = 1000
        nk = 5 
                     lnorm1 = Array{Float64, 2}(undef, N, nk)
                     lnorm2 = Array{Float64, 2}(undef, N, nk)
                     lnorm3 = Array{Float64, 2}(undef, N, nk)
                     lall = Array{Float64, 2}(undef, N, nk)
                     lik = Array{Float64, 2}(undef, N, nk)
                     lpm = Array{Float64, 2}(undef, N, nk)

    lτ = log.(τ)
    for i in 1:N  
        lnorm1[i,:] = lognormpdf.(Y1[i],μ[1,:],σ[1,:])
        lnorm2[i,:] = lognormpdf.(Y2[i],μ[2,:],σ[2,:])
        lnorm3[i,:] = lognormpdf.(Y3[i],μ[3,:],σ[3,:])
    end
    lall = lτ + lnorm1 + lnorm2 + lnorm3
    lik = lik + logsumexp.(lall)

    for i in 1:N  
        τ[i,:] = exp.(lall[i,:]) ./ sum(exp.(lall[i,:]))
    end
    τ
    return(τ,lik)
    
end

function Maximization(Y1, 
                      Y2,
                      Y3,
                      μ,
                      σ,
                      τ, 
                      N,
                      nk)
                      
    DY1 = kron(Y1,1* Matrix(I, nk, nk))
    DY2 = kron(Y2,1* Matrix(I, nk, nk))
    DY3 = kron(Y3,1* Matrix(I, nk, nk))

    Dkj1 = kron(fill(1, (N,1)), 1* Matrix(I, nk, nk))
    Dkj2 = kron(fill(1, (N,1)), 1* Matrix(I, nk, nk))  
    Dkj3 = kron(fill(1, (N,1)), 1* Matrix(I, nk, nk))

    Ω1 = vec(reshape(τ, size(DY1, 1),1))
    Ω2 = vec(reshape(τ, size(DY2, 1),1))
    Ω3 = vec(reshape(τ, size(DY3, 1),1))

    WLS1 = WLS(Dkj1, DY1, Ω1)
    WLS2 = WLS(Dkj2, DY2, Ω2)
    WLS3 = WLS(Dkj3, DY3, Ω3)

    for i in 1:nk  
        μ[1,i] = 1/N*sum(Dkj1[:,1].*WLS1[i,i])
        σ[1,i] = sqrt(1/N*sum((DY1[:,i] - Dkj1[:,1].*WLS1[i,i]).^2))
    end
    for i in 1:nk  
        μ[2,i] = 1/N*sum(Dkj2[:,1].*WLS2[i,i])
        σ[2,i] = sqrt(1/N*sum((DY2[:,i] - Dkj2[:,1].*WLS2[i,i]).^2))
    end
    for i in 1:nk  
        μ[3,i] = 1/N*sum(Dkj3[:,1].*WLS3[i,i])
        σ[3,i] = sqrt(1/N*sum((DY3[:,i] - Dkj3[:,1].*WLS3[i,i]).^2))
    end

    return(μ,σ)
end

function ExpectationMaximization(Y1,Y2,Y3,μ,σ,π,N,nk)

        Y1 = Y1_test
        Y2 = Y2_test
        Y3 = Y3_test 
        μ = μ_test
        σ = σ_test
        π = π_test
        N = 1000
        nk = 5 

    τ = repeat(π,N)
        τ = reshape(τ, N, nk)

    exp_update = Expectation(Y1,Y2,Y3,μ,σ,τ,N,nk)

    τ_update = exp_update[1]
    lik_update = sum(exp.(exp_update[2]))

    max_update = Maximization(Y1,Y2,Y3,μ,σ,τ_update,N,nk)

    μ_update = max_update[1]
    σ_update = max_update[2]

    iter = 0
    lik_diff = 0
  

    while lik_diff > 1e-2 || iter < 10
    
    
        lik = lik_update
    
        exp_update = Expectation(Y1,Y2,Y3,μ_update,σ_update,τ_update,N,nk)
        
        τ_update = exp_update[1]
        lik_update = sum(exp.(exp_update[2]))

        max_update = Maximization(Y1,Y2,Y3,μ_update,σ_update,τ_update,N,nk)

        μ_update = max_update[1]
        σ_update = max_update[2]
    
        
        lik_diff = lik_update - lik
        
        
        
    
        iter = iter + 1
        
    
    end

    return(τ_update, lik_update, μ_update, σ_update, iter)

end


# %%

N = 1000
nk = 5
tau = Array{Float64, 2}(undef, N, nk)
lpm = Array{Float64, 2}(undef, N, nk)



# Now I'm going to randomly generate my Y's from a mixture of Gaussians with a
π0 = fill(1/(2*nk), ((2*nk),1))
# This gives the initial values for the std dev of each Gaussian model
σ0 = repeat(fill(1, ((2*nk),1)), 3)
    σ0 = reshape(σ0,3,(2*nk))
# This gives the initial values for the means of each Gaussian model
μ0 = repeat(collect(1:(2*nk)),inner = 3)
   μ0 = reshape(μ0, 3,(2*nk))
# Simulating the Sample from the true DGP constructed from π0, μ0 and σ0
Y = rand(MvNormal(μ0[1,:], I), 3*N)
    Y = π0' * Y
    Y = Y'

Y1 = Y[1:N]
Y2 = Y[N+1:2*N]
Y3 = Y[2*N+1:3*N]

# This denotes the initial probability assigned to each Gaussian model
π_test = fill(1/nk, (nk,1))
# This gives the initial values for the std dev of each Gaussian model
σ = Array{Float64}(repeat(fill(1, (nk,1)),3))
σ = reshape(σ,3,nk)
# This gives the initial values for the means of each Gaussian model
μ = Array{Float64}(repeat(collect(1:nk),inner = 3))
μ = reshape(μ, 3,nk)



    τ = repeat(π_test,N)
        τ = reshape(τ, N, nk)

    
    exp_update = Expectation(Y1,Y2,Y3,μ,σ,τ,N,nk)

    τ_update = exp_update[1]
    lik_update = sum(exp.(exp_update[2]))

    max_update = Maximization(Y1,Y2,Y3,μ,σ,τ_update,N,nk)

    μ_update = max_update[1]
    σ_update = max_update[2]

    iter = 0
    lik_diff = 0
  

    # while lik_diff > 1e-9 || iter < 10
    
    
        lik = lik_update
    
    
        
        lik_update = sum(ExpectationMaximization(Y1,Y2,Y3,μ_update,σ_update,τ,N,nk)[1])
        μ_update = ExpectationMaximization(Y1,Y2,Y3,μ,σ,π,N,nk)[2]
        σ_update = ExpectationMaximization(Y1,Y2,Y3,μ,σ,π,N,nk)[3]
        
    
        iter = iter + 1
        iter
    
    # end
    
  μ_update 

Expectation(Y1, Y2, Y3, μ, σ, π, N, nk)

x = ExpectationMaximization(Y1,Y2,Y3,μ,σ,π,N,nk)
