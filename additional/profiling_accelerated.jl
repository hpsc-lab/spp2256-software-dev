### A Pluto.jl notebook ###
# v0.20.18

using Markdown
using InteractiveUtils

# ╔═╡ 30fef252-8f23-11f0-055e-b1fb9ce7d6fb
begin 
	using CUDA
	using KernelAbstractions
	using Atomix
	using BenchmarkTools
	using Adapt
	using NVTX
	using ImageShow
end

# ╔═╡ e991afa2-8e37-4f8c-987a-bf9daf24f6cd
using CUDA: i32

# ╔═╡ 0c2e2176-2d7f-4e2c-9405-462f20c2ca7f
using Random

# ╔═╡ 3acfbfad-1233-4e08-a7d4-66508dad8c0d
versioninfo()

# ╔═╡ be442c44-9544-4439-a778-d6ea6d60dd3e
md"""

## A first GPU kernel
"""

# ╔═╡ 962a10e7-0a06-43dd-af30-fdc891e1287b
function copy_cpu!(A, B)
  for I in 1:length(A)
    @inbounds A[I] = B[I]
  end
end

# ╔═╡ 9668c885-143a-41dc-afd5-3ed802edf245
@kernel function copy_kernel!(A, B)
  I = @index(Global)
  @inbounds A[I] = B[I]
end

# ╔═╡ 548a2642-cbe9-41e5-aa2e-749290fb0026
function copy_ka!(A, B)
  backend = get_backend(A)
  @assert size(A) == size(B)
  @assert get_backend(B) == backend

  kernel = copy_kernel!(backend)
  kernel(A, B, ndrange = length(A))
  return
end

# ╔═╡ 6a13741e-3ba6-4a77-b7b3-2e751e332ef5
function copy_kernel_cuda!(A, B)
  I = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
  if I <= length(A)
      @inbounds A[I] = B[I]
  end
  return nothing
end

# ╔═╡ ff1f0e27-9f09-4560-a29b-dccb76770f43
function copy_cuda!(A, B)
  kernel = @cuda launch=false copy_kernel_cuda!(A, B)
  config = launch_configuration(kernel.fun)
  threads = min(length(A), config.threads)
  blocks = cld(length(A), threads)

  kernel(A, B; threads, blocks)
end

# ╔═╡ d2a95316-a182-415a-8b41-15bbe2c07664
B = rand(64_000);

# ╔═╡ c5943384-1ed8-4ff9-848c-1c6da05cb4e3
let
  A = similar(B)
  copy_cpu!(A, B)
  @assert A == B
end

# ╔═╡ 7f1734de-fef1-47f9-ac91-18867f0b5073
md"""
Julia GPU ecosystem follows the motto: Compute follows Data

So let's move our data to the GPU
"""

# ╔═╡ baaa778f-0896-4392-81d5-f3b3203958be
d_B = adapt(CuArray, B);

# ╔═╡ 9a7d6edb-96e2-44be-a196-e340330ba0b3
typeof(d_B)

# ╔═╡ 8926d1a4-ca39-4276-938d-37af3e67e3e4
let
  d_A = similar(d_B)
  copy_cuda!(d_A, d_B)
  @assert d_A == d_B
end

# ╔═╡ 8fc9ee68-ebbf-4585-a1e8-1a05d436ac93
md"""
Note that Julia GPU Ecosystem, synchronizes the GPU on access. So we are launchign two GPU kernels here, first the copy, then the comparision and they are both executing asynchronously, but ordered with respect to each other.

We then "wait" for the result of the comparision.
"""

# ╔═╡ edf1bd01-2f7f-467e-ac7c-0c86161a5330
@benchmark copy_cuda!(d_A, $d_B) setup=(d_A = similar(d_B))

# ╔═╡ 62eb782a-d900-416d-8791-cb664c14502b
CUDA.@profile let
  d_A = similar(d_B)
  for _ in 1:10
    copy_cuda!(d_A, d_B)
  end
end

# ╔═╡ 8be22bb2-5a71-436c-a641-2c01e9435207
md"""
So there seems to be a discrepancy between the measurement of `@benchmark` and `CUDA.@profile`, `@benchmark` seems to vastly over-estimate the performance of the GPU code. To remedy this we need to include a synchronization operation with benchmarking.
"""

# ╔═╡ 956e72d5-2b0e-4d57-8ca1-74a8cc5e87a8
@benchmark CUDA.@sync(copy_cuda!(d_A, $d_B)) setup=(d_A = similar(d_B))

# ╔═╡ 862df26a-05e0-4d01-8cbb-6441b3bbb1e6
md"""
With KernelAbstractions we can now write code that is portable and can be used both for data that resides on the CPU as well as the GPU, therefore implementing the "Compute follows Data" paradigm.
"""

# ╔═╡ 87c50921-e91e-40fe-aa9c-cb810732bc74
let
  A = similar(B)
  copy_ka!(A, B)
  @assert A == B
end

# ╔═╡ 7973ca15-2339-4917-8c03-3716c09e9d25
let
  d_A = similar(d_B)
  copy_ka!(d_A, d_B)
  @assert d_A == d_B
end

# ╔═╡ ed289720-b625-454f-b5be-c1d37c51b68a
@benchmark copy_ka!(A, $B) setup=(A = similar(B))

# ╔═╡ c1a3907e-5f71-4cd1-ab2e-1ecc44ef8e92
@benchmark CUDA.@sync(copy_ka!(d_A, $d_B)) setup=(d_A = similar(d_B))

# ╔═╡ d83203f5-3284-4acb-8d90-216aa43e0339
CUDA.@profile let
  d_A = similar(d_B)
  for _ in 1:10
    copy_ka!(d_A, d_B)
  end
end

# ╔═╡ 19f60fb3-934a-4414-b037-2cccfe663f47
md"""
We can see that KernelAbstractions is a bit slower than pure CUDA, and that is partially expected due to some convenience functionality.
"""

# ╔═╡ e41b7693-d1a9-4807-a6a6-641b3a58c787
@kernel unsafe_indices=true function copy_kernel_faster!(A, B)
  N = @uniform prod(@groupsize())
  I = (@index(Group, Linear)-1i32) * N + @index(Local, Linear)
  if I <= length(A)
    @inbounds A[I] = B[I]
  end
end

# ╔═╡ 5a48d053-64e5-410f-9146-0aaeef9df514
CUDA.@profile let
  d_A = similar(d_B)
  for _ in 1:10
    copy_kernel_faster!(CUDABackend())(d_A, d_B, ndrange=length(d_A))
  end
end

# ╔═╡ 13c91be9-cb41-4c13-928c-6844064285e5
md"""
## A more compilcated kernel -- transpose
"""

# ╔═╡ 2b8dafa0-9558-4912-a900-fb2cead352bd
const nreps = 3
const N = 2048
const T = Float32

const TILE_DIM = 32
const BLOCK_ROWS = 8

# ╔═╡ a1bfe2e2-e921-4ba3-9563-0dd0b7bbd26f
md"""
### Naive kernels
"""

# ╔═╡ c8ce4e2e-670e-4ba3-8576-0b5be256b795
@kernel function simple_copy_kernel!(output, @Const(input))
    I, J = @index(Global, NTuple)
    @inbounds output[I, J] = input[I, J]
end

# ╔═╡ d409ebc8-8864-442b-a91b-e651bf7957bf
@kernel function simple_transpose_kernel!(output, @Const(input))
    I, J = @index(Global, NTuple)
    @inbounds output[J, I] = input[I, J]
end

# ╔═╡ e8a1c8b7-c425-42c6-8ce4-772564118561
md"""
### Using localmemory
"""

# ╔═╡ 1e94f48a-8e1e-49bb-b936-21795ca195c4
@kernel unsafe_indices = true function lmem_copy_kernel!(
        output, @Const(input),
        ::Val{BANK} = Val(1),
    ) where {BANK}
    I, J = @index(Global, NTuple)
    i, j = @index(Local, NTuple)

    N = @uniform @groupsize()[1]
    M = @uniform @groupsize()[2]

    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(output) (N + BANK, M)

    @inbounds tile[i, j] = input[I, J]

    @synchronize

    @inbounds output[I, J] = tile[i, j]
end

# ╔═╡ f5c41ddc-b6b4-4dd9-b7e0-29b395859003
@kernel unsafe_indices = true function lmem_transpose_kernel!(
        output, @Const(input),
        ::Val{BANK} = Val(1),
    ) where {BANK}
    gi, gj = @index(Group, NTuple)
    i, j = @index(Local, NTuple)

    N = @uniform @groupsize()[1]
    M = @uniform @groupsize()[2]

    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(output) (N + BANK, M)

    # Manually calculate global indexes
    # Later on we need to pivot the group index
    I = (gi - 1) * N + i
    J = (gj - 1) * M + j

    @inbounds tile[i, j] = input[I, J]

    @synchronize

    # Pivot the group index
    I = (gj - 1) * M + i
    J = (gi - 1) * N + j

    @inbounds output[I, J] = tile[j, i]
end

# ╔═╡ 3bdc33cd-8b8a-4b91-a11b-13a23981f7cf
### Local Memory + process multiple elements per lane

# ╔═╡ d1a3bf73-99c0-4d05-a6c0-980d1dec75ae
import KernelAbstractions.Extras: @unroll

# ╔═╡ 0bdcd901-ce4f-4fd7-96fb-e60a988cdab2
@kernel unsafe_indices=true function coalesced_copy_kernel!(
        output, @Const(input),
        ::Val{BANK} = Val(1),
    ) where {BANK}
    gi, gj = @index(Group, NTuple)
    i, j = @index(Local, NTuple)

    TILE_DIM = @uniform @groupsize()[1]
    BLOCK_ROWS = @uniform @groupsize()[2]

    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(output) (TILE_DIM + BANK, TILE_DIM)

    # Can't use @index(Global), because we use a smaller ndrange
    I = (gi - 1) * TILE_DIM + i
    J = (gj - 1) * TILE_DIM + j

    @unroll for k in 0:BLOCK_ROWS:(TILE_DIM - 1)
        @inbounds tile[i, j + k] = input[I, J + k]
    end

    @synchronize

    @unroll for k in 0:BLOCK_ROWS:(TILE_DIM - 1)
        @inbounds output[I, J + k] = tile[i, j + k]
    end
end

# ╔═╡ c2468ec4-f7cd-4e14-a7ad-10f4855e81ba
@kernel unsafe_indices = true function coalesced_transpose_kernel!(
        output, @Const(input),
        ::Val{BANK} = Val(1),
    ) where {BANK}
    gi, gj = @index(Group, NTuple)
    i, j = @index(Local, NTuple)

    TILE_DIM = @uniform @groupsize()[1]
    BLOCK_ROWS = @uniform @groupsize()[2]

    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(output) (TILE_DIM + BANK, TILE_DIM)

    # Can't use @index(Global), because we use a smaller ndrange
    I = (gi - 1) * TILE_DIM + i
    J = (gj - 1) * TILE_DIM + j

    @unroll for k in 0:BLOCK_ROWS:(TILE_DIM - 1)
        @inbounds tile[i, j + k] = input[I, J + k]
    end

    @synchronize

    # Transpose block offsets
    I = (gj - 1) * TILE_DIM + i
    J = (gi - 1) * TILE_DIM + j

    @unroll for k in 0:BLOCK_ROWS:(TILE_DIM - 1)
        @inbounds output[I, J + k] = tile[j + k, i]
    end
end

# ╔═╡ 58d5e165-4aa2-46b6-803d-91125985eb76
### Benchmark harness

# ╔═╡ ffb87594-d922-49cb-892e-d425c860daef
backend = CUDABackend()

# ╔═╡ 0268c55f-cbe0-4631-90b5-6e3555cd0946
CUDA.@profile for block_dims in ((TILE_DIM, TILE_DIM), (TILE_DIM * TILE_DIM, 1), (1, TILE_DIM * TILE_DIM))
    for (name, kernel) in (
            ("copy", simple_copy_kernel!(backend, block_dims)),
            ("transpose", simple_transpose_kernel!(backend, block_dims)),
        )
        NVTX.@range "Simple $name $block_dims" let
            input = rand!(allocate(backend, T, N, N))
            output = similar(input)

            # compile kernel
            kernel(output, input, ndrange = size(output))
            for rep in 1:nreps
                kernel(output, input, ndrange = size(output))
            end
            KernelAbstractions.synchronize(backend)
        end
    end
end

# ╔═╡ 88c246ed-63ab-4d62-a095-7b045aa12f9e
# Benchmark localmem
CUDA.@profile for (name, kernel) in (
        ("copy", lmem_copy_kernel!(backend, (TILE_DIM, TILE_DIM))),
        ("transpose", lmem_transpose_kernel!(backend, (TILE_DIM, TILE_DIM))),
    )
    for bank in (true, false)
        NVTX.@range "Localmem $name ($TILE_DIM, $TILE_DIM) bank=$bank" let
            input = rand!(allocate(backend, T, N, N))
            output = similar(input)

            # compile kernel
            kernel(output, input, Val(Int(bank)), ndrange = size(output))
            for rep in 1:nreps
                kernel(output, input, Val(Int(bank)), ndrange = size(output))
            end
            KernelAbstractions.synchronize(backend)
        end
    end
end

# ╔═╡ badb5f10-0dde-44d0-b7d0-424cdad44dc7
# Benchmark localmem + multiple elements per lane
CUDA.@profile for (name, kernel) in (
        ("copy", coalesced_copy_kernel!(backend, (TILE_DIM, BLOCK_ROWS))),
        ("transpose", coalesced_transpose_kernel!(backend, (TILE_DIM, BLOCK_ROWS))),
    )
    for bank in (true, false)
        NVTX.@range "Localmem + multiple elements $name ($TILE_DIM, $BLOCK_ROWS) bank=$bank" let
            input = rand!(allocate(backend, T, N, N))
            output = similar(input)

            # We want a number of blocks equivalent to (TILE_DIM, TILE_DIM)
            # but our blocks are (TILE_DIM, BLOCK_ROWS) so we need to remove
            # a factor from the size of the array otherwise we get to many blocks
            block_factor = div(TILE_DIM, BLOCK_ROWS)
            ndrange = (N, div(N, block_factor))

            # compile kernel
            kernel(output, input, Val(Int(bank)), ndrange = ndrange)
            for rep in 1:nreps
                kernel(output, input, Val(Int(bank)), ndrange = ndrange)
            end
            KernelAbstractions.synchronize(backend)
        end
    end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
Atomix = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
ImageShow = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
NVTX = "5da4648a-3479-48b8-97b9-01cb529c0a1f"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Adapt = "~4.3.0"
Atomix = "~1.1.2"
BenchmarkTools = "~1.6.0"
CUDA = "~5.8.3"
ImageShow = "~0.3.8"
KernelAbstractions = "~0.9.38"
NVTX = "~1.0.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.6"
manifest_format = "2.0"
project_hash = "38d471ed51658d6dccf21ec0cf517fde0173bf91"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

    [deps.AbstractFFTs.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "f7817e2e585aa6d924fd714df1e2a84be7896c60"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.3.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "29bb0eb6f578a587a49da16564705968667f5fa8"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.2"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "3b642331600250f592719140c60cf12372b82d66"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.5.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "e38fbc49a620f5d0b660d7f543db1009fe0f8336"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Compiler_jll", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "DataFrames", "ExprTools", "GPUArrays", "GPUCompiler", "GPUToolbox", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "StaticArrays", "Statistics", "demumble_jll"]
git-tree-sha1 = "27f69b3923e58730f0a71396070e9114fc0bba40"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "5.8.3"

    [deps.CUDA.extensions]
    ChainRulesCoreExt = "ChainRulesCore"
    EnzymeCoreExt = "EnzymeCore"
    SparseMatricesCSRExt = "SparseMatricesCSR"
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.CUDA.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    SparseMatricesCSR = "a0a7dd2c-ebf4-11e9-1f05-cf50bc540ca1"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.CUDA_Compiler_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "8c4f340dd6501a93c4b99b690797772e4a203099"
uuid = "d1e2174e-dfdc-576e-b43e-73b79eb1aca8"
version = "0.2.1+0"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e6a1d9f5518122c186fd27786b61d2053cfa1b0c"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "13.0.1+0"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "f9a521f52d236fe49f1028d69e549e7f2644bb72"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "1.0.0"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "e24c6de116c0735c37e83b8bc05ed60d4d359693"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.19.1+0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "a656525c8b46aa6a1c76891552ed5381bb32ae7b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.30.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

    [deps.ColorTypes.weakdeps]
    StyledStrings = "f489334b-da3d-4c2e-b8f0-e476e12c162b"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "0037835448781bb46feb39866934e243886d756a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "c967271c27a95160e30432e011b58f42cd7501b5"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "6c72198e6a101cccdd4c9731d3985e904ba26037"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.1"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "b66970a70db13f45b7e57fbda1736e1cf72174ea"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "ScopedValues", "Serialization", "Statistics"]
git-tree-sha1 = "0b6c695edc49ed61dfa322639354ff5a6e2c9a32"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.2.4"

    [deps.GPUArrays.extensions]
    JLD2Ext = "JLD2"

    [deps.GPUArrays.weakdeps]
    JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "PrecompileTools", "Preferences", "Scratch", "Serialization", "TOML", "Tracy", "UUIDs"]
git-tree-sha1 = "eb1e212e12cc058fa16712082d44be499d23638c"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "1.6.1"

[[deps.GPUToolbox]]
deps = ["LLVM"]
git-tree-sha1 = "5bfe837129bf49e2e049b4f1517546055cc16a93"
uuid = "096a3bc2-3ced-46d0-87f4-dd12716f4bfc"
version = "0.3.0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaNVTXCallbacks_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "af433a10f3942e882d3c671aacb203e006a5808f"
uuid = "9c1d0b0a-7046-5b2e-a33f-ea22f176ac7e"
version = "0.2.1+0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "83c617e9e9b02306a7acab79e05ec10253db7c87"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.38"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "9c7c721cfd800d87d48c745d8bfb65144f0a91df"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.4.2"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "2ea068aac1e7f0337d381b0eae3110581e3f3216"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.37+2"

[[deps.LLVMLoopInfo]]
git-tree-sha1 = "2e5c102cfc41f48ae4740c7eca7743cc7e7b75ea"
uuid = "8b046642-f1f6-4319-8d3c-209ddc03c586"
version = "1.0.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.LibTracyClient_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d2bc4e1034b2d43076b50f0e34ea094c2cb0a717"
uuid = "ad6e5548-8b26-5c9f-8ef3-ef0ad883f3a5"
version = "0.9.1+6"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NVTX]]
deps = ["Colors", "JuliaNVTXCallbacks_jll", "Libdl", "NVTX_jll"]
git-tree-sha1 = "6b573a3e66decc7fc747afd1edbf083ff78c813a"
uuid = "5da4648a-3479-48b8-97b9-01cb529c0a1f"
version = "1.0.1"

[[deps.NVTX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "af2232f69447494514c25742ba1503ec7e9877fe"
uuid = "e98f9f5b-d649-5603-91fd-7774390e6439"
version = "3.2.2+0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "dbe5fd0b334694e905cb9fda73cd8554333c46e2"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.1"

[[deps.RandomNumbers]]
deps = ["Random"]
git-tree-sha1 = "c6ec94d2aaba1ab2ff983052cf6a606ca5985902"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.6.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "b8693004b385c842357406e3af647701fe783f98"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.15"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Tracy]]
deps = ["ExprTools", "LibTracyClient_jll", "Libdl"]
git-tree-sha1 = "73e3ff50fd3990874c59fef0f35d10644a1487bc"
uuid = "e689c965-62c8-4b79-b2c5-8359227902fd"
version = "0.1.6"

    [deps.Tracy.extensions]
    TracyProfilerExt = "TracyProfiler_jll"

    [deps.Tracy.weakdeps]
    TracyProfiler_jll = "0c351ed6-8a68-550e-8b79-de6f926da83c"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "b13c4edda90890e5b04ba24e20a310fbe6f249ff"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.0"
weakdeps = ["LLVM"]

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.demumble_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6498e3581023f8e530f34760d18f75a69e3a4ea8"
uuid = "1e29f10c-031c-5a83-9565-69cddfc27673"
version = "1.3.0+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═30fef252-8f23-11f0-055e-b1fb9ce7d6fb
# ╠═3acfbfad-1233-4e08-a7d4-66508dad8c0d
# ╠═be442c44-9544-4439-a778-d6ea6d60dd3e
# ╠═962a10e7-0a06-43dd-af30-fdc891e1287b
# ╠═9668c885-143a-41dc-afd5-3ed802edf245
# ╠═548a2642-cbe9-41e5-aa2e-749290fb0026
# ╠═e991afa2-8e37-4f8c-987a-bf9daf24f6cd
# ╠═6a13741e-3ba6-4a77-b7b3-2e751e332ef5
# ╠═ff1f0e27-9f09-4560-a29b-dccb76770f43
# ╠═d2a95316-a182-415a-8b41-15bbe2c07664
# ╠═c5943384-1ed8-4ff9-848c-1c6da05cb4e3
# ╠═7f1734de-fef1-47f9-ac91-18867f0b5073
# ╠═baaa778f-0896-4392-81d5-f3b3203958be
# ╠═9a7d6edb-96e2-44be-a196-e340330ba0b3
# ╠═8926d1a4-ca39-4276-938d-37af3e67e3e4
# ╠═8fc9ee68-ebbf-4585-a1e8-1a05d436ac93
# ╠═edf1bd01-2f7f-467e-ac7c-0c86161a5330
# ╠═62eb782a-d900-416d-8791-cb664c14502b
# ╠═8be22bb2-5a71-436c-a641-2c01e9435207
# ╠═956e72d5-2b0e-4d57-8ca1-74a8cc5e87a8
# ╠═862df26a-05e0-4d01-8cbb-6441b3bbb1e6
# ╠═87c50921-e91e-40fe-aa9c-cb810732bc74
# ╠═7973ca15-2339-4917-8c03-3716c09e9d25
# ╠═ed289720-b625-454f-b5be-c1d37c51b68a
# ╠═c1a3907e-5f71-4cd1-ab2e-1ecc44ef8e92
# ╠═d83203f5-3284-4acb-8d90-216aa43e0339
# ╠═19f60fb3-934a-4414-b037-2cccfe663f47
# ╠═e41b7693-d1a9-4807-a6a6-641b3a58c787
# ╠═5a48d053-64e5-410f-9146-0aaeef9df514
# ╠═13c91be9-cb41-4c13-928c-6844064285e5
# ╠═2b8dafa0-9558-4912-a900-fb2cead352bd
# ╠═a1bfe2e2-e921-4ba3-9563-0dd0b7bbd26f
# ╠═c8ce4e2e-670e-4ba3-8576-0b5be256b795
# ╠═d409ebc8-8864-442b-a91b-e651bf7957bf
# ╠═e8a1c8b7-c425-42c6-8ce4-772564118561
# ╠═1e94f48a-8e1e-49bb-b936-21795ca195c4
# ╠═f5c41ddc-b6b4-4dd9-b7e0-29b395859003
# ╠═3bdc33cd-8b8a-4b91-a11b-13a23981f7cf
# ╠═d1a3bf73-99c0-4d05-a6c0-980d1dec75ae
# ╠═0bdcd901-ce4f-4fd7-96fb-e60a988cdab2
# ╠═c2468ec4-f7cd-4e14-a7ad-10f4855e81ba
# ╠═58d5e165-4aa2-46b6-803d-91125985eb76
# ╠═0c2e2176-2d7f-4e2c-9405-462f20c2ca7f
# ╠═ffb87594-d922-49cb-892e-d425c860daef
# ╠═0268c55f-cbe0-4631-90b5-6e3555cd0946
# ╠═88c246ed-63ab-4d62-a095-7b045aa12f9e
# ╠═badb5f10-0dde-44d0-b7d0-424cdad44dc7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
