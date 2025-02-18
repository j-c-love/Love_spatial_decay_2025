using Turing, Optim, DataFrames, CSV
using NumericalIntegration, DelimitedFiles, BenchmarkTools
using DifferentialEquations, StatsPlots, LinearAlgebra
using Random
using Interpolations
using Plots
using Statistics
#theme(:dracula)
theme(:dark)
Random.seed!(14);

t = collect(0:20:1200)

# ODE function
function smFISH(du, u, p, t)
    F, D, γ = p
    du[1] = γ*F(t) .- D*u[1]
    return nothing
end

# Function to compute integral
function calculate_integrand(D, γ, F, t)
    γ*integrate(t, F.*exp.(D*(t.-t[end])))
end

# Turing model
@model function smfish(F, t)
    # sample degradation
    #D ~ filldist(Gamma(2, 2), 5)
    D ~ filldist(truncated(Normal(0, 1)), 5)
    #m ~ Normal(0, sqrt(s²))
    γ ~ InverseGamma(2, 3)
    σ ~ InverseGamma(2, 3)
    #integrand_all = []
    integrand_all = Vector{Float64}(undef, 0)
    for i in 1:lastindex(D)
        integrand = [calculate_integrand(D[i], γ, F[i][j,:], t) for j in 1:lastindex(D)]
        integrand_all = vcat(integrand_all, integrand)
    end
    m ~ MvNormal(integrand_all, σ^2*I)
end

# Generate synthetic data
function generate_synthetic_data(Array_F, Array_D, generative_γ, initial_timepoint)
    simulated_final_points = zeros(length(Array_D))
    for i in 1:length(Array_D)
        itp = linear_interpolation(0:60, Array_F[i,:])
        D = Array_D[i]
        p = [itp, D, generative_γ]
        tspan = (0.0, 60)
        prob = ODEProblem(smFISH, initial_timepoint, tspan, p)
        sol = solve(prob, Tsit5())

        simulated_final_points[i] = sol(60)[1]
    end
    #println(simulated_final_points)
    noisy_data = simulated_final_points .+ 0.1*mean(simulated_final_points)*randn(length(Array_D))
    #println(noisy_data)
    noise = noisy_data - simulated_final_points
    return(simulated_final_points, noisy_data)
end 

F_data = DataFrame(CSV.File(""; header=false))
F_data = Array(F_data)
m_data = DataFrame(CSV.File(""; header=false))
m_data = Array(m_data)

u0 = [0.] # initial condition = 0 everywhere

F_data
F1_data = F_data[1:5,:]
F2_data = F_data[6:10,:]
F3_data = F_data[11:15,:]
F4_data = F_data[16:20,:]
F5_data = F_data[21:25,:]

F_data_arrays = Array[F1_data, F2_data, F3_data, F4_data, F5_data]

# Infer parameters from the model
model = smfish(F_data_arrays, t) 
chn = sample(model | (; m = m_data), NUTS(), MCMCThreads(), 10000, 4)
get(chn; section=:parameters)
plot(chn, show=true)
df = DataFrame(chn)
println(describe(df))

samples = Array(chn)
samples = round.(samples, digits = 1)

D1 = samples[:,1]
D2 = samples[:,2]
D3 = samples[:,3]
D4 = samples[:,4]
D5 = samples[:,5]
D_posteriors = Array[D1, D2, D3, D4, D5]
HL_posteriors = log(2)./(D_posteriors)
D_posteriors[1]
HL_posteriors[1]

posterior_γ = samples[:,6]
posterior_σ = samples[:,7]

plot_font = "Arial"
default(fontfamily=plot_font,
        linewidth=2, framestyle=:box, label=nothing, grid=false)

l = @layout [a; b; c; d; e]
p1 = plot(histogram(D_posteriors[1], label="D]"))
plot!([mode(D_posteriors[1])],linetype=:vline,widths=3, label="mode", color=:tempo)
plot!([median(D_posteriors[1])],linetype=:vline,widths=3, label="median", color=:magma)
p2 = plot(histogram(D_posteriors[2], label="D2"))
plot!([mode(D_posteriors[2])],linetype=:vline,widths=3, label="mode", color=:tempo)
plot!([median(D_posteriors[2])],linetype=:vline,widths=3, label="median", color=:magma)
p3 = plot(histogram(D_posteriors[3], label="D3"))
plot!([mode(D_posteriors[3])],linetype=:vline,widths=3, label="mode", color=:tempo)
plot!([median(D_posteriors[3])],linetype=:vline,widths=3, label="median", color=:magma)
p4 = plot(histogram(D_posteriors[4], label="D4"))
plot!([mode(D_posteriors[4])],linetype=:vline,widths=3, label="mode", color=:tempo)
plot!([median(D_posteriors[4])],linetype=:vline,widths=3, label="median", color=:magma)
p5 = plot(histogram(D_posteriors[5], label="D5"))
plot!([mode(D_posteriors[5])],linetype=:vline,widths=3, label="mode", color=:tempo)
plot!([median(D_posteriors[5])],linetype=:vline,widths=3, label="median", color=:magma)
plot(p1, p2, p3, p4, p5, layout = l)
plot!(size=(500,600))

mode(D_posteriors[1])
mode(D_posteriors[2])
mode(D_posteriors[3])
mode(D_posteriors[4])
mode(D_posteriors[5])

median(D_posteriors[1])
median(D_posteriors[2])
median(D_posteriors[3])
median(D_posteriors[4])
median(D_posteriors[5])

theme(:mute)

plot_font = "Arial"
default(fontfamily=plot_font,
        linewidth=2, framestyle=:box, label=nothing, grid=false)

l = @layout [a; b; c; d; e]
p1 = plot(histogram(D_posteriors[1], label="D1", color="thistle", linecolor="plum3", bins = 0:0.1:5))
plot!([mode(D_posteriors[1]), mode(D_posteriors[1])], [0, 3200],label="mode")
p2 = plot(histogram(D_posteriors[2], label="D2", color="thistle", linecolor="plum3", bins = 0:0.1:5))
plot!([mode(D_posteriors[2]), mode(D_posteriors[2])], [0, 4000],label="mode")
p3 = plot(histogram(D_posteriors[3], label="D3", color="thistle", linecolor="plum3", bins = 0:0.1:5))
plot!([mode(D_posteriors[3]), mode(D_posteriors[3])], [0, 9000],label="mode")
p4 = plot(histogram(D_posteriors[4], label="D4", color="thistle", linecolor="plum3", bins = 0:0.1:5))
plot!([mode(D_posteriors[4]), mode(D_posteriors[4])], [0, 3200],label="mode")
p5 = plot(histogram(D_posteriors[5], label="D5", color="thistle", linecolor="plum3", bins = 0:0.1:5))
plot!([mode(D_posteriors[5]), mode(D_posteriors[5])], [0, 4000],label="mode")

plot(p1, p2, p3, p4, p5, layout = l)
plot!(size=(500,800))

plot(chn, show=true, size=(840, 1100))

mode(posterior_σ)
mode(posterior_γ)

# for writing
CSV.write("",df)