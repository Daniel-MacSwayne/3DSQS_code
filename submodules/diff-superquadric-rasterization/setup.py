# setup.py: Build configuration for the diff-superquadric-rasterization CUDA extension.
#
# Mirrors the setup.py in diff-gaussian-rasterization but compiles the superquadric
# kernel files. No third-party math library needed (all matrix math is inline CUDA).
#
# Install with:
#   conda run -n 3DSQS pip install -e .

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# Source files to compile:
#   rasterizer_impl.cu   — orchestration (preprocess → sort → render)
#   forward.cu           — preprocessCUDA + renderCUDA kernels
#   backward.cu          — renderBackwardCUDA + preprocessBackwardCUDA kernels
#   rasterize_superquadrics.cu — PyTorch tensor wrapper
#   ext.cpp              — pybind11 bindings
sources = [
    os.path.join("cuda_rasterizer", "rasterizer_impl.cu"),
    os.path.join("cuda_rasterizer", "forward.cu"),
    os.path.join("cuda_rasterizer", "backward.cu"),
    "rasterize_superquadrics.cu",
    "ext.cpp",
]

setup(
    name="diff_superquadric_rasterization",
    packages=["diff_superquadric_rasterization"],
    ext_modules=[
        CUDAExtension(
            name="diff_superquadric_rasterization._C",
            sources=sources,
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "-arch=sm_86",          # RTX 4090 (Ada Lovelace)
                    "--use_fast_math",
                    "-Xcompiler", "-fPIC",
                ],
                "cxx": ["-O3", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
