# %%
using Random, 
      Distributions,
      LinearAlgebra,
      StatsBase,
      StatsPlots,
      StatFiles

Random.seed!(123)

function sim_data(k,n, T; doplot = false)
    # true values
    μ = randn(k) * 3
    # for i in 1:k
    #     μ[i] = (μ[i]+2*i)*(-1)^i
    # end

    σ = abs.(randn(k))./2
    p = rand(k)
    p = p / sum(p)
    K_draw = sample(1:k, Weights(p), n, replace = true)
    Y = Matrix{Float64}(undef, n, T)
    for t in 1:T
        Y[:, t] = rand.(Normal.(μ[K_draw], σ[K_draw])) 
    end

    return Dict(:y => Y, :μ => μ, :σ => σ, :p => p, :K_draw => K_draw)
end


# Pretty much directly inherited from Thomas Wiemann
mutable struct GaussMix
    μ # component means
    σ # component standard devs
    probs # component probabilities
    probs_post # posterior component probabilities
    y # data
    K # number of components
    function GaussMix(y, K)
        # Initialize component parameters randomly
        nobs = length(y)
        μ = rand(Normal(0, std(y)), K)
        σ = std(y) * ones(K) ./ K
        probs = rand(Dirichlet(ones(K)))
        probs_post = zeros((nobs , K))
        # Construct object
        new(μ, σ, probs, probs_post, y, K)
    end
end


# This too 
function par_fit!(m:: GaussMix; max_iter = 1000, tol = 1e-2)
    # Get values from m
    μ = m.μ; σ = m.σ
    probs = m.probs; probs_post = m.probs_post
    K = m.K; nobs = length(m.y)
    
    # Run EM algorithm
    likeli = zeros((nobs , K))
    log_l = zeros(max_iter)
    for j in 1:max_iter
        # E-step
        for k in 1:K
            likeli[:, k] = pdf.(Normal(μ[k], σ[k]), m.y) .* probs[k]
        end
        probs_post = likeli ./ repeat(mapslices(sum , likeli , dims = 2)', K)'
        
        # M-step
        for k in 1:K
            sum_prob_post = sum(probs_post[:, k])
            probs[k] = mean(probs_post[:, k])
            μ[k] = sum(m.y .* probs_post[:, k]) ./ sum_prob_post
            σ[k] = sum((m.y .- μ[k]).^2 .* probs_post[:, k]) ./ sum_prob_post
            σ[k] = sqrt(σ[k])
        end
        # Compute current value of the likelihood
        mixture_j = MixtureModel(Normal , [(μ[k], σ[k]) for k in 1:K], probs)
        log_l[j] = sum(logpdf.(mixture_j, m.y))
        
    end
    # Export values to m
    m.μ = μ; m.σ = σ
    m.probs = probs; m.probs_post = probs_post
    return nothing
end

# EM funcs

nk_test = 3
n_test = 2000

fake_sim_data = sim_data(nk_test, n_test, 3)
Y_test = fake_sim_data[:y]
Y1_test = Y_test[:, 1]
Y2_test = Y_test[:, 2]
Y3_test = Y_test[:, 3]

mix_model_test = GaussMix(Y1_test[:], nk_test)
par_fit!(mix_model_test)

hcat(sort(mix_model_test.μ), sort(fake_sim_data[:μ]))

using Plots
p_data = density(Y1_test);
p_estimated = plot(Normal.(mix_model_test.μ, mix_model_test.σ));
p_true = plot(Normal.(fake_sim_data[:μ], fake_sim_data[:σ]));
l = @layout [a ; b ; c]
plot(
    p_data,
    p_estimated,
    p_true,
    layout = l
)


using DataFrames, GLM
stat_df = DataFrame(load("data/AER_2012_1549_data/output/data4estimation.dta"))


stat_df

fit = lm(
    @formula(log_y ~ year + marit + state_st),
    stat_df
)


using Pkg
Pkg.add(url="https://github.com/EdJeeOnGitHub/Jeeves")

regression_df = select(
    stat_df,
    ["log_y",
     "year",
     "person",
     "marit",
     "state_st"]
)
using Jeeves
clean_regression_df = regression_df[completecases(regression_df), :]

year_dummy_matrix = Jeeves.dummy_matrix(clean_regression_df, "year")
state_dummy_matrix = Jeeves.dummy_matrix(clean_regression_df, "state_st")


model = Jeeves.OLSModel(
    clean_regression_df.log_y,
    hcat(
        clean_regression_df[!, ["marit"]],
        year_dummy_matrix[!, 2:end],
        state_dummy_matrix
    )
)

year_dummy_matrix[!, 2:end]

jeeves_fit = Jeeves.fit(model)
fit_resid = jeeves_fit.modelfit.resid
Jeeves.tidy(jeeves_fit)

clean_regression_df[!, "y_resid"] = fit_resid
clean_regression_df
using Chain, DataFramesMeta
psid_y_data = @chain clean_regression_df begin
    sort([:person, :year])
    groupby([:person])
    @transform(
        :y_resid_lead_1 = lead(:y_resid),
        :y_resid_lead_2 = lead(:y_resid, 2),
        :y_resid_lead_3 = lead(:y_resid, 3)
    )
    @select(:y_resid, :y_resid_lead_1, :y_resid_lead_2, :y_resid_lead_3)
end


psid_y_data = psid_y_data[completecases(psid_y_data[!, [:y_resid,
                                                        :y_resid_lead_1,
                                                        :y_resid_lead_2]]), :]

psid_mix_models_3 = GaussMix.([psid_y_data.y_resid, 
                               psid_y_data.y_resid_lead_1, 
                               psid_y_data.y_resid_lead_2], 3)
psid_mix_models_4 = GaussMix.([psid_y_data.y_resid, 
                               psid_y_data.y_resid_lead_1, 
                               psid_y_data.y_resid_lead_2], 4)
psid_mix_models_5 = GaussMix.([psid_y_data.y_resid, 
                               psid_y_data.y_resid_lead_1, 
                               psid_y_data.y_resid_lead_2], 5)
par_fit!.(vcat(psid_mix_models_3, psid_mix_models_4, psid_mix_models_5))


psid_models = vcat(
    psid_mix_models_3,
    psid_mix_models_4,
    psid_mix_models_5
)
using StatsPlots

function plot_mixture(mix_model)
    mm_model = MixtureModel(
            Normal.(
                mix_model.μ,
                mix_model.σ
            ),
            mix_model.probs
        )
    p_estim = plot(
        mm_model
    )
    p_data = density(mix_model.y, label = "Y Observed")
    p_stacked_dens = plot(x -> pdf(mm_model, x), minimum(mix_model.y), maximum(mix_model.y),
                            label = "Y Estim" )
    p_all = plot(
        p_data,
        p_stacked_dens,
        p_estim,
        layout = @layout [a ; b ; c])
    return p_all
end



plot_mixture(psid_mix_models_3[1])
plot_mixture(psid_mix_models_4[1])
plot_mixture(psid_mix_models_5[1])

plot_mixture(psid_mix_models_3[2])
plot_mixture(psid_mix_models_4[2])
plot_mixture(psid_mix_models_5[2])

plot_mixture(psid_mix_models_3[3])
plot_mixture(psid_mix_models_4[3])
plot_mixture(psid_mix_models_5[3])


function gen_mm_model(m::GaussMix)

    mm_model = MixtureModel(
            Normal.(
                m.μ,
                m.σ
            ),
            m.probs
        )
    return mm_model
end


mm_models = gen_mm_model.(
    vcat(psid_mix_models_3,
         psid_mix_models_4,
         psid_mix_models_5)
)



test_mm = mm_models[1]
quantile(psid_mix_models_3[1].y, 0.0:0.01:1.0)

quantile(test_mm, 0.0:0.01:1.0)


function qq_mixtures(estim_m::GaussMix)
    mix_model = gen_mm_model(estim_m)

    q_data = quantile(estim_m.y, 0.0:0.01:1.0)
    q_pdf = quantile(mix_model, 0.0:0.01:1.0)
    df = DataFrame(
        :q_data => q_data,
        :q_pdf => q_pdf
    )
    return df
end

qq_dfs = qq_mixtures.(psid_models)

function plot_qq(qq_mix)

    plot(qq_mix.q_data, qq_mix.q_pdf, seriestype = :scatter,
         xlabel = "Data", ylabel = "Mixture PDF", label = "QQ")
    Plots.abline!(1, 0, label = "Theoretical") 
    Plots.xlims!(minimum(qq_mix.q_data[2:end]), maximum(qq_mix.q_data[1:(end-1)]))
end

plot(
    plot_qq(qq_dfs[1]),
    plot_qq(qq_dfs[2]),
    plot_qq(qq_dfs[3]),
    plot_qq(qq_dfs[4])
    )




fd_df = function (df, ρ)
    
    new_df = @chain df begin
        @transform(
            :y_fd_3 = :y_resid_lead_3 - ρ*:y_resid_lead_2,
            :y_fd_2 = :y_resid_lead_2 - ρ*:y_resid_lead_1,
            :y_fd_1 = :y_resid_lead_1 - ρ*:y_resid
        )
    end
    new_df = new_df[completecases(new_df), :]
    return new_df
end


fd_psid_df = fd_df(psid_y_data, 0.6)



fd_mix_models_3 = GaussMix(fd_psid_df.y_fd_1, 3)
fd_mix_models_4 = GaussMix(fd_psid_df.y_fd_1, 4)
fd_mix_models_5 = GaussMix(fd_psid_df.y_fd_1, 5)

fd_mix_models = vcat(
    fd_mix_models_3,
    fd_mix_models_4,
    fd_mix_models_5
)


par_fit!.(fd_mix_models)

plot_mixture(fd_mix_models[1])
plot_mixture(fd_mix_models[2])
plot_mixture(fd_mix_models[3])

qq_fd_dfs = qq_mixtures.(fd_mix_models)

plot(
    plot_qq(qq_fd_dfs[1]),
    plot_qq(qq_fd_dfs[2]),
    plot_qq(qq_fd_dfs[3])
)

# We didn't do MLE sorry