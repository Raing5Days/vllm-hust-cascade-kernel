#!/usr/bin/env python
# coding=utf-8

"""python api for ascend_kernel."""

import os
from configparser import ConfigParser
from pathlib import Path

import setuptools
from setuptools import find_namespace_packages
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution
from torch_npu.utils.cpp_extension import NpuExtension


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""

    def has_ext_modules(self):
        return True


class Build(build_ext, object):

    def run(self):
        self.build_lib = os.path.relpath(os.path.join(BASE_DIR, "build"))
        self.build_temp = os.path.relpath(os.path.join(BASE_DIR, "build/temp"))
        self.library_dirs.append(os.path.relpath(os.path.join(BASE_DIR, "build/lib")))
        super(Build, self).run()


WORKING_DIR = Path(__file__).resolve().parent
config = ConfigParser()
config.read(WORKING_DIR / "ascend_kernel" / "config.ini")
_version = config.get("global", "version")


def _stage_license() -> str:
    """Copy the repository LICENSE next to setup.py so the wheel embeds it.

    CANN OSL 2.0 §3.3 requires redistribution of the Software (or a derivative
    work of it -- our `.so` instantiates the vendored `catlass` template tree, see
    ../../../NOTICE) to carry a copy of the agreement.  The canonical text lives
    at the repository root; staging a copy here keeps that single source of truth
    while making every packaging path self-sufficient -- `build.sh`, a direct
    `python setup.py bdist_wheel`, or a PEP 517 build all find it the same way.
    """
    for candidate in (WORKING_DIR / "LICENSE", *(
        parent / "LICENSE" for parent in WORKING_DIR.parents
    )):
        if candidate.is_file():
            target = WORKING_DIR / "LICENSE"
            if candidate != target:
                target.write_bytes(candidate.read_bytes())
            return "LICENSE"
    raise SystemExit(
        f"ERROR: no LICENSE found above {WORKING_DIR} -- the wheel would "
        "redistribute catlass-derived code without the CANN OSL 2.0 text."
    )


_license_file = _stage_license()


setuptools.setup(
    name="ascend-kernel",
    version=_version,
    description="python api for ascend_kernel",
    packages=find_namespace_packages(exclude=("tests*",)),
    ext_modules=[NpuExtension("ascend_kernel._C", sources=[])],
    # The .so instantiates the vendored CANN `catlass` template tree, so this is
    # a CANN Open Software derivative work: CANN OSL 2.0 §2.1 permits distribution
    # for Huawei AI processor systems and §3.3 requires the agreement to travel
    # with it -- `license_files` embeds the staged text in the wheel.
    license="CANN Open Software License Agreement Version 2.0",
    license_files=[_license_file],
    python_requires=">=3.7",
    package_data={"ascend_kernel": ["lib/**", "VERSION"]},
)
