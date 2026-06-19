module ChebyshevFilter

import LinearAlgebra
import KPM
import SparseArrays

"""
Generates function to calculate Chebyshev interpolation coefficients for 
sqrt(h/(1+ kappa - x)). `n` is the order of the expansion.
"""
function generate_c(n, kappa, h)
    j = Array(0:n) 
    xj = (j .+ 1/2) .* pi ./ (n + 1)

    function c(k)
        return 1/(n+1)*sum(sqrt.(h ./ (1 .+ kappa .- cos.(xj))) .* cos.(k .* xj))
    end

    return c
end

"""
Calculates approximate projection of `ket` onto the subspace spanned by the 
eigenvectors of `x` with eigenvalues less than `a` based on Allen-Zhu and Li 
2016.
# Arguments
- x: matrix to base eigenvalue projection on
- NC: number of chebyshev iterations to compute.
- a: eigenvalue cutoff
- delta: x should have no eigenvalues within +/- `delta` of `a`. A larger 
    `delta` will result in faster convergence.
- ket: matrix whose columns are vectors that the projection will be applied to
"""
function step_function(x, NC, a, delta, ket)
    h = 2/((1 + abs(a))^2 - delta^2)
    kappa = h*delta^2
    y = (1 + kappa)*LinearAlgebra.I - h*(x-a*LinearAlgebra.I)^2
    g = threaded_kpm_expansion(y, generate_c(NC, kappa, h), NC, ket,
                        LinearAlgebra.I; kernel = (n, NC) -> 1)
    return 1/2*(-(x - a*LinearAlgebra.I) * g + LinearAlgebra.I*ket)
end 

"""
Calculates the approximate projection of `ket` onto subspace spanned by 
eigenvectors of H with eigenvalues between `a` and `b` based on Allen-Zhu and
Li 2016.

# Arguments
- H: matrix to base the projection on
- NC: number of Chebyshev iterations to calculate
- a: lower bound for eigenvalues in subspace
- b: upper bound for eigenvalues in subspace
- delta: H should not have any eigenvalues within +/- `delta` of `a` or `b`
- ket: matrix whose columns are vectors that the projection will be applied to
"""
function fast_chebyshev_indicator(H, NC, a, b, delta, ket)
    step_b = step_function(H, NC, b, delta, ket)
    step_a = step_function(H, NC, a, delta, ket)
    return step_b .- step_a
end

"""
Calculates coefficients of Chebyshev expansion for indicator function of the
interval (a, b).
"""
function indicator_coefficients(n, a, b)
    if n == 0
        return (acos(a) - acos(b))/pi
    else
        return (sin(n*acos(a)) - sin(n*acos(b)))/(n*pi)
    end
end

"""
Calculates the approximate projection of `ket` onto subspace spanned by 
eigenvectors of H with eigenvalues between `a` and `b` usig a Chebyshev
expansion of the indicator function.
"""
function chebyshev_indicator(H, NC, a, b, ket)
    return threaded_kpm_expansion(H, n -> indicator_coefficients(n, a, b), NC, 
                                  ket, LinearAlgebra.I)
end

#to make combatable with `sketch_projection`
function chebyshev_indicator(H, NC, a, b, delta, ket)
    return chebyshev_indicator(H, NC, a, b, ket)
end

"""
Calculates Chebyshev expansion of <bra| sum(alpha(n) * Tn(H)) |ket>, where
Tn is the nth Chebyshev polynomial of the 1st kind. See Weisse et al. 2006 for 
details.
# Arguments
- H: matrix argument of the Chebyshev polynomials
- alpha: alpha(n) is the nth Chebyshev expansion coefficient
- NC: Order of Chebyshev expansion
- ket: Chebyshev expansion is right-multiplied by ket
- bra: Chebyshev expansion is left-multiplied by bra
- kernel: kernel(n, NC) is the nth kernel coefficient for expansion order NC
"""
function kpm_expansion(H, alpha, NC::Int64, ket, bra;
        kernel=KPM.JacksonKernel) 
    g = n -> kernel(n, NC)
    T = LinearAlgebra.I*ket
    T_next = H*ket

    result = g(0)*alpha(0)*T + 2*g(1)*alpha(1)*T_next
    temp1 = similar(T)
    temp2 = similar(T)

    for n in 2:NC-1
        #T, T_next = T_next, 2*H*T_next - T
        #result += 2*g(n)*alpha(n)*T_next
        temp1 .= T_next
        LinearAlgebra.mul!(temp2, H, T_next)
        T_next .= 2 .* temp2 .- T
        T .= temp1 #TODO: I think I can just switch T and temp w/out .
        result .+= 2 .* g(n) .* alpha(n) .* T_next
    end
    
    return bra*result
end


"""
Calculates Chebyshev expansion of sum(alpha(n, E) * Tn(H)) |ket>, where
Tn is the nth Chebyshev polynomial of the 1st kind. The expansion is
broadcasted over each value of E, where E[i] is an additional parameter for
alpha
# Arguments
- H: matrix argument of the Chebyshev polynomials
- E: vector of 
- alpha: alpha(n, E) is the nth Chebyshev expansion coefficient
- NC: Order of Chebyshev expansion
- ket: Chebyshev expansion is right-multiplied by ket
- bra: Chebysheve expansion is left-multiplied by bra
- kernel: kernel(n, NC) is the nth kernel coefficient for expansion order NC
"""
function kpm_expansion(H, E, alpha, NC::Int64, ket;
        kernel=KPM.JacksonKernel) 
    g = n -> kernel(n, NC)
    T = LinearAlgebra.I*ket
    T_next = H*ket
    
    #moments = zeros(ComplexF64, size(ket)..., size(E, 1))
    moments = zeros(typeof(ket), size(ket)..., size(E, 1))
    E = Array(reshape(E, 1, 1, :))
    #result = g(0)*alpha(0)*T + 2*g(1)*alpha(1)*T_next
    #moments .+= (g(0)* T) .* alpha.(0, E)
    #moments .+= (2*g(1)*T_next) .* alpha.(1, E)
    moments .+= (g(0)* T) .* alpha.(0, E)
    moments .+= (2*g(1)*T_next) .* alpha.(1, E)
    temp1 = similar(T)
    temp2 = similar(T)
    #alphas = zeros(ComplexF64, 1, 1, size(E,3))
    alphas = zeros(typeof(ket), 1, 1, size(E,3))

    #a = (n, E) -> alpha(n, E)
    for n in 2:NC-1
        #T, T_next = T_next, 2*H*T_next - T
        #result += 2*g(n)*alpha(n)*T_next
        temp1 .= T_next
        LinearAlgebra.mul!(temp2, H, T_next)
        T_next .= 2 .* temp2 .- T
        T, temp1 = temp1, T
        #result .+= 2 .* g(n) .* alpha(n) .* T_next
        
        alphas .= alpha.(n, E)
        #alphas .= a.(n, E)
        moments .+= (2 .* g(n)  .* T_next) .* alphas

    end
    
    return moments
end

"""
KPM expansion with columns split between threads
"""
function threaded_kpm_expansion(H, alpha, NC, ket, bra;
        kernel=KPM.JacksonKernel)
    N_partitions = Threads.nthreads()
    D, l = size(ket)
    result = zeros(ComplexF64, D, l)
    N_per_partition = div(l, N_partitions)
    Threads.@threads for i in 1:Threads.nthreads()
        indices = 1 + (i-1)*N_per_partition:1:i*N_per_partition
        result[:,indices] = kpm_expansion(H, alpha, NC, ket[:,indices], LinearAlgebra.I,
                                     kernel=kernel)
    end
    return bra*result
end

#TODO: check if this is faster/better
function kpm_expansion_new(H, alpha, NC, ket, bra;
        kernel=KPM.JacksonKernel) 
    g = n -> kernel(n, NC)
    T = LinearAlgebra.I*ket #TODO: can store T in ket instead
    @show sizeof(T)
    T_next = H*ket
    @show sizeof(T_next)

    result = g(0)*alpha(0)*T + 2*g(1)*alpha(1)*T_next
    for n in 2:NC-1
        #T = 2*H*T_next - T
        mul!(T, H, T_next, 2, -1)
        T, T_next = T_next, T
        result .+= 2 .* g(n) .* alpha(n) .* T_next
    end

    return bra*result
end

"""
Computes the approxmate indicator function of H defined above applied to a
standard complex Guassian matrix using multithreading.
# Arguments
- H: matrix to base the projection on
- a: lower bound for eigenvalues in subspace
- b: upper bound for eigenvalues in subspace
- delta: H should not have any eigenvalues within +/- `delta` of `a` or `b`
- l: number of columns of the Gaussian matrix
- NC: number of Chebyshev iterations to compute
"""
function sketch_projection(H::AbstractMatrix{ComplexF64}, a::Float64, 
        b::Float64, delta::Float64, l::Int, NC::Int; filter=fast_chebyshev_indicator)
    D = size(H)[1]
    #N_partitions = MPI.Comm_size(comm)
    @show N_partitions = Threads.nthreads()
    N_per_partition = div(l, N_partitions)
    #ket = Array{ComplexF64}(undef, D, l)
    ket = zeros(ComplexF64, D, l)
    P = zeros(ComplexF64, D, l)

    println("calculating expansion...")
    flush(stdout)
    Threads.@threads for i in 1:Threads.nthreads()
        indices = 1 + (i-1)*N_per_partition:1:i*N_per_partition
        ket[:,indices] = randn(ComplexF64, D, N_per_partition)
        #P[:,indices] = filter(H, NC, a, b, delta, ket[:,indices])
    end

    @time P = filter(H, NC, a, b, delta, ket)
    return P
end

"""
Gets column space of a matrix via QR decomposition
# Arguments
- P: matrix to get the column space of
- tol: the jth column of Q is discarded if Rjj <= tol

# Returns
- matrix whose columns form an orthonormal basis for the column space of P
"""
function get_column_space(P::AbstractMatrix{ComplexF64}; tol=1e-5)
    Q, R = LinearAlgebra.qr(P, LinearAlgebra.ColumnNorm())
    Q = Matrix(Q)
    @show sort(abs.(LinearAlgebra.diag(R)))
    Q = Q[:,abs.(LinearAlgebra.diag(R)) .> tol]
    @show size(Q) 
    @show size(P)
    return Q
end

#TODO: should be able to specify delta for each side
"""
Finds an orthonormal basis for the subspace spanned by eigenvectors of H
with eigenvalues between `a` and `b` by applying the random rangefinder
algorithm in Halko et al. 2011 to a polynomial expansion to the subspace
projection matrix based on Allen-Zhu and Li 2016.

# Arguments
- H: matrix to base the projection on
- a: lower bound for eigenvalues in subspace
- b: upper bound for eigenvalues in subspace
- delta: H should not have any eigenvalues within +/- `delta` of `a` or `b`
- l: number of columns of the Gaussian matrix
- NC: number of Chebyshev iterations to compute
- tol: see `get_column_space` 
# Returns
- Q: matrix whose columns form an orthonormal basis for the subspace
    (the projection matrix P can be calculated as P = Q*adjoint(Q))
"""
function random_rangefinder(H::AbstractMatrix{ComplexF64}, a::Float64, 
        b::Float64, delta::Float64, l::Int, NC::Int; tol=1e-5, 
        filter=fast_chebyshev_indicator)

    P = sketch_projection(H, a, b, delta, l, NC; filter=filter)
    @show maximum(abs.(P))

    Q = get_column_space(P; tol=tol)

    return Q
end

end
