"""Build script."""
from pybind11.setup_helpers import Pybind11Extension, build_ext

extensions = [
    Pybind11Extension(
        "sopht_mpi.numeric.immersed_boundary_ops._cosine_kernel_helper",
        sources=["sopht_mpi/numeric/immersed_boundary_ops/CosineKernelHelper.cpp"],
        language="c++",
        cxx_std=17,
        extra_compile_args=["-O3", "-march=native"],
    )
]

def build(setup_kwargs):
    setup_kwargs.update({"ext_modules": extensions, "cmdclass": {"build_ext": build_ext}})
