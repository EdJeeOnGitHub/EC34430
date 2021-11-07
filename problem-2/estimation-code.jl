module lewd # labour estimators, wrongly done
    
using Distributions
using LinearAlgebra
using Infiltrator
using StatsBase
using DataFrames
using Plots
using CategoricalArrays
using ShiftedArrays
using Chain
using DataFramesMeta
using LightGraphs
# using TikzGraphs
using Optim
using Random

# Find all movers defined as those with max spell > 0
function find_movers(df::DataFrame)
    move_df = @chain df begin
       groupby(:w_id)
       transform(:spell => maximum) 
       subset(:spell_maximum => x -> x .> 0)
    end

   return move_df 
end


# Get the next firm the mover is moving to
# by leading the firm column and extracting 
# just the firm and next firm as one observation
function find_firm_links(mover_df::DataFrame)
    firm_link_df = @chain mover_df begin
        sort([:w_id, :t])
        groupby(:w_id)
        transform(:j => lead => :j_next)
        transform([:j, :j_next] => .==)
        subset(:j_j_next_BroadcastFunction => x -> x .== false, skipmissing = true)
        select(:j, :j_next)
    end
    return firm_link_df
end



function create_M_flows(link_df)
    M = @chain link_df begin
        groupby([:j, :j_next])
        combine(nrow => :count)
        unique()
    end
    return M
end


# %%


function create_adjacency_matrix(firm_link_df::DataFrame, df::DataFrame; count = false)
    adjacency_matrix = zeros(Int, maximum(df.j), maximum(df.j))
    if count == false
        firm_link_df[!, "count"] .= 1
    end
    for firm_a in unique(df.j), firm_b in unique(df.j)
        subset_a_df = firm_link_df[(firm_link_df.j .== firm_a) .& (firm_link_df.j_next .== firm_b), :]
        if size(subset_a_df)[1] != 0
            adjacency_matrix[firm_b, firm_a] = subset_a_df.count[1]
        end
    end
    return adjacency_matrix
end

function create_fixed_point_matrices(M_df, df)
    sum_flows = @chain M_df begin
        groupby(:j)
        combine(:count => sum)
        sort(:j)
        unique()
    end
    S_kk = Diagonal(sum_flows.count_sum)
    M_0 = create_adjacency_matrix(M_df, df, count = true)
    return S_kk, M_0
end


function create_fixed_point_matrices(df)
    link_df = find_firm_links(find_movers(df))
    M_df = create_M_flows(link_df)

    M_0 = create_adjacency_matrix(M_df, df, count = true)
    S_kk = Diagonal(sum(M_0, dims = 1)[:])
    return S_kk, M_0
end

function create_adjacency_matrix(firm_link_df::DataFrame, df::DataFrame)
    adjacency_matrix = zeros(Int, maximum(df.j), maximum(df.j))
    for firm_a in unique(df.j), firm_b in unique(df.j)
        subset_a_df = firm_link_df[(firm_link_df.j .== firm_a) .& (firm_link_df.j_next .== firm_b), :]
        if size(subset_a_df)[1] != 0
            adjacency_matrix[firm_a, firm_b] = 1
            adjacency_matrix[firm_b, firm_a] = 1
        end
    end
    return adjacency_matrix
end


function create_connected_df(df::DataFrame)
    move_df = find_movers(df)
    firm_df = find_firm_links(move_df)
    adjacency_matrix = create_adjacency_matrix(firm_df, move_df)


    simple_graph = SimpleGraph(adjacency_matrix);

    connected_network = connected_components(simple_graph)

    connected_set = connected_network[1]

    println("We have only $(length(connected_set)) firms fully connected $(length(connected_set)/maximum(df.j)) of the market...") # Double check we might be wrong...

    df_connected = df[in(connected_set).(df.j),:]
    return df_connected
end



function estimate_rank(S_kk, M_0; tol= 1e-6)
    N_connected = size(S_kk, 1)
    S_inv = inv(S_kk)
    initial_V_EE = fill(0.5, N_connected)
    lhs = S_inv * M_0 * exp.(initial_V_EE)
    rhs = exp.(initial_V_EE)
    i = 0
    max_diff = Inf
    while max_diff > tol
        i += 1
        rhs = lhs
        lhs = S_inv*M_0 * rhs
        max_diff = maximum(abs.(lhs .- rhs))
        println("max diff: $max_diff")
    end
    return rhs
end
end