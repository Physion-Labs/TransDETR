"""Pip-installable packaging for TransDETR (Physion-Labs fork).

Upstream (bingshuai2019/TransDETR) ships the two CUDA ops with their own
sub-setups meant for in-place builds; this root setup builds everything into
one installable `trans_detr` package so consumers can depend on a pinned git
URL instead of a checkout + PYTHONPATH.

Extension module names are load-bearing:

* ``MultiScaleDeformableAttention`` — TOP-LEVEL, exactly as the code imports
  it (``import MultiScaleDeformableAttention as MSDA``).
* ``trans_detr.models.Rotated_ROIAlign.rotated_roi`` — namespaced, matching
  ``import trans_detr.models.Rotated_ROIAlign.rotated_roi as _C``.

Build requirements: torch with CUDA available at build time — install with
``pip install --no-build-isolation`` in an env that already has torch
(standard practice for the Deformable-DETR family; build isolation would hide
the env's torch from this script).
"""

import glob
import os

from setuptools import find_namespace_packages, setup

ROOT = os.path.dirname(os.path.abspath(__file__))


def _cuda_extension(name: str, ext_root: str):
    from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension

    # Build only needs nvcc + CUDA headers (CUDA_HOME). It does NOT need a
    # GPU to be visible at install time — nvcc cross-compiles for the arches
    # listed in TORCH_CUDA_ARCH_LIST (or torch's defaults). Requiring a GPU
    # here blocks containerized CI builds for no real reason; the loadable
    # check happens naturally when the .so is imported at runtime.
    if CUDA_HOME is None:
        raise NotImplementedError(
            f"CUDA toolkit required to build {name} (CUDA_HOME is None). "
            "Install on a machine with nvcc + CUDA headers present. "
            "A GPU at install time is NOT required."
        )

    src = os.path.join(ROOT, ext_root, "src")
    sources = (glob.glob(os.path.join(src, "*.cpp"))
               + glob.glob(os.path.join(src, "cpu", "*.cpp"))
               + glob.glob(os.path.join(src, "cuda", "*.cu")))
    return CUDAExtension(
        name,
        sorted(sources),
        include_dirs=[src],
        define_macros=[("WITH_CUDA", None)],
        extra_compile_args={
            "cxx": [],
            "nvcc": [
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        },
    )


def get_ext_modules():
    return [
        _cuda_extension("MultiScaleDeformableAttention", "trans_detr/models/ops"),
        _cuda_extension("trans_detr.models.Rotated_ROIAlign.rotated_roi",
                        "trans_detr/models/Rotated_ROIAlign"),
    ]


def get_cmdclass():
    from torch.utils.cpp_extension import BuildExtension

    return {"build_ext": BuildExtension}


setup(
    name="trans-detr",
    version="1.0.0",
    description="TransDETR: video text spotting (Physion-Labs installable fork)",
    url="https://github.com/Physion-Labs/TransDETR",
    packages=find_namespace_packages(
        include=["trans_detr", "trans_detr.*"],
        exclude=["*build*", "*egg-info*"],
    ),
    python_requires=">=3.9",
    # Inference-path runtime deps (empirically determined — the import chain of
    # models/datasets/util pulls these at module import). torch/torchvision are
    # deliberately NOT declared: environments own their torch builds, and a
    # resolver-driven torch reinstall is never wanted here.
    install_requires=[
        "numpy",
        "opencv-python",
        "Pillow",
        "shapely",
        "pyclipper",
        "Polygon3",   # `import Polygon` in datasets/data_tools.py (GPL upstream dep)
    ],
    ext_modules=get_ext_modules(),
    cmdclass=get_cmdclass(),
)
