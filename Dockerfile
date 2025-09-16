# syntax=docker/dockerfile:1
FROM julia:1.11.6
# FROM ghcr.io/juliahpc/derse25-workshop:main

# Install git, for use within Codespaces
RUN /bin/sh -c 'export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y git \
    && apt-get --purge autoremove -y \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/*'

# Docker is awful and doesn't allow conditionally setting environment variables in a decent
# way, so we have to keep an external script and source it every time we need it.
COPY julia_cpu_target.sh /julia_cpu_target.sh

RUN julia --color=yes -e 'using InteractiveUtils; versioninfo()'

# Instantiate Julia project
RUN mkdir -p /root/.julia/environments/v1.11
COPY Project.toml  /root/.julia/environments/v1.11/Project.toml
# Preinstall some common packages across all notebooks
RUN . /julia_cpu_target.sh && julia --color=yes -e 'using Pkg; Pkg.add(["BenchmarkTools", "CairoMakie", "CUDA", "ChunkSplitters", "KernelAbstractions", "oneAPI", "OhMyThreads", "Pluto", "PlutoUI", "ThreadPinning"])'

# Copy notebooks
COPY lectures/01_getting_started_with_julia.jl /root/lectures/01_getting_started_with_julia.jl
COPY lectures/02_pde_parallelism.jl /root/lectures/02_pde_parallelism.jl
COPY lectures/03_GPU_computing.jl /root/lectures/03_GPU_computing.jl
COPY lectures/04_interop.jl /root/lectures/04_interop.jl
COPY exercises/01_newton.jl /root/exercises/01_newton.jl
COPY exercises/02_shared_memory.jl /root/exercises/02_shared_memory.jl
COPY additional/performance_engineering.jl /root/additional/performance_engineering.jl
