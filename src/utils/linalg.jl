const AbstractVectorOrNTuple{T} = Union{AbstractVector{T},NTuple{N,T}} where {N}

sum_to_one!(x) = ldiv!(sum(x), x)

mynonzeros(x::AbstractArray) = x
mynonzeros(x::AbstractSparseArray) = nonzeros(x)

mynnz(x::AbstractArray) = length(mynonzeros(x))

elementwise_log(x::AbstractArray) = log.(x)

function elementwise_log(A::SparseMatrixCSC)
    return SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, log.(A.nzval))
end

"""
    mul_rows_cols!(B, l, A, r)

Perform the in-place operation `B .= l .* A .* transpose(r)`.
"""
function mul_rows_cols!(
    B::AbstractMatrix, l::AbstractVector, A::AbstractMatrix, r::AbstractVector
)
    B .= l .* A .* transpose(r)
    return B
end

function mul_rows_cols!(
    B::SparseMatrixCSC, l::AbstractVector, A::SparseMatrixCSC, r::AbstractVector
)
    @argcheck axes(A, 1) == eachindex(r)
    @argcheck axes(A, 2) == eachindex(l)
    @argcheck size(A) == size(B)
    @argcheck nnz(B) == nnz(A)
    Brv = rowvals(B)
    Bnz = nonzeros(B)
    Anz = nonzeros(A)
    @simd for j in axes(B, 2)
        @argcheck nzrange(B, j) == nzrange(A, j)
        @simd for k in nzrange(B, j)
            i = Brv[k]
            Bnz[k] = l[i] * Anz[k] * r[j]
        end
    end
    return B
end

"""
    argmaxplus_transmul!(y, ind, A, x)

Perform the in-place multiplication `transpose(A) * x` _in the sense of max-plus algebra_, store the result in `y`, and store the index of the maximum for each component of `y` in `ind`.
"""
function argmaxplus_transmul!(
    y::AbstractVector{R},
    ind::AbstractVector{<:Integer},
    A::AbstractMatrix,
    x::AbstractVector,
) where {R}
    @argcheck axes(A, 1) == eachindex(x)
    @argcheck axes(A, 2) == eachindex(y)
    fill!(y, typemin(R))
    fill!(ind, 0)
    @simd for j in axes(A, 2)
        @simd for i in axes(A, 1)
            z = A[i, j] + x[i]
            if z > y[j]
                y[j] = z
                ind[j] = i
            end
        end
    end
    return y
end

function argmaxplus_transmul!(
    y::AbstractVector{R},
    ind::AbstractVector{<:Integer},
    A::SparseMatrixCSC,
    x::AbstractVector,
) where {R}
    @argcheck axes(A, 1) == eachindex(x)
    @argcheck axes(A, 2) == eachindex(y)
    Anz = nonzeros(A)
    Arv = rowvals(A)
    fill!(y, typemin(R))
    fill!(ind, 0)
    @simd for j in axes(A, 2)
        @simd for k in nzrange(A, j)
            i = Arv[k]
            z = Anz[k] + x[i]
            if z > y[j]
                y[j] = z
                ind[j] = i
            end
        end
    end
    return y
end

function argmaxplus_transmul!(
    y::AbstractMatrix{R},
    ind::AbstractMatrix{<:Integer},
    A::AbstractMatrix,
    x::AbstractMatrix,
    r::AbstractMatrix{<:Integer},
    k::Integer,
) where {R}
    @argcheck axes(A, 1) == axes(x, 1)
    @argcheck axes(A, 2) == axes(y, 1)
    fill!(y, typemin(R))
    fill!(ind, 0)
    fill!(r, 0)
    @simd for j in axes(A, 2)
        z = Tuple{R, Int}[]
        @simd for i in axes(A, 1)
            append!(z, collect(zip(A[i,j] .+ x[i,:], repeat([i], k))))
        end
        sort!(unique!(z), by = x -> x[1], rev = true)
        aux = length(z) ≥ k ? k : length(z)
        rank = Dict{Int, Int}(1:size(A, 2) .=> 0)
        y[j, 1:aux], ind[j, 1:aux] = map(x -> getfield.(z[1:aux], x), fieldnames(eltype(z)))
        foreach((i,l) -> r[j,l] = (rank[i] += 1) , ind[j, 1:aux], 1:aux)
    end
    return y
end