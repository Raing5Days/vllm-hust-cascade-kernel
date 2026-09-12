# host side files
FILE(GLOB OP_SRCS
    ${PROJECT_OP_SRC_BASE}/register.cpp
    ${PROJECT_OP_SRC_BASE}/ops/lse_merge/op_host/lse_merge.cpp
    ${PROJECT_OP_SRC_BASE}/ops/fa_fp32_stage1/op_host/fa_fp32_stage1.cpp
    ${PROJECT_OP_SRC_BASE}/ops/add_rms_norm_stats/op_host/add_rms_norm_stats.cpp
    ${PROJECT_OP_SRC_BASE}/ops/bw_probe/op_host/bw_probe.cpp
    ${PROJECT_OP_SRC_BASE}/ops/f3_floor/op_host/f3_floor.cpp
    )

# set the library output dir to the python dir for wheel package build
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/ascend_kernel/ascend_kernel/lib)
if(BUILD_CATLASS_MODULE)
    list(APPEND OP_SRCS
        ${PROJECT_OP_SRC_BASE}/ops/catlass/op_host/catlass_matmul_basic.cpp
    )
endif()

# set the so name
set(OP_PLUGIN_NAME ascend_kernel)

# kernel side files without workspace.
# lse_merge keeps its own pristine target (exact original configuration): sharing
# a target with the catlass mix kernel changed its build/stub flow (two-binary
# registration) and broke its fp32 path on device - see plan-20260903 log.
ascendc_library(no_workspace_kernel STATIC
    ${PROJECT_OP_SRC_BASE}/ops/lse_merge/op_kernel/kernel_lse_merge.cpp
)

# fa_fp32_stage1: catlass mix kernel (AIC+AIV) in its own target; the vendored
# catlass headers live in third_party/catlass (see its README.md); debug.hpp
# pulls <acl/acl.h> hence the CANN include dir.
ascendc_library(no_workspace_kernel_fa STATIC
    ${PROJECT_OP_SRC_BASE}/ops/fa_fp32_stage1/op_kernel/kernel_fa_fp32_stage1.cpp
)
ascendc_include_directories(no_workspace_kernel_fa PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/catlass/include
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include
)

# add_rms_norm_stats: pure AIV (vector) kernel, two entries (bf16/fp16). Own
# target for the same reason as lse_merge below (the mix-kernel target changes
# the stub/registration flow); no catlass dependency.
ascendc_library(no_workspace_kernel_norm STATIC
    ${PROJECT_OP_SRC_BASE}/ops/add_rms_norm_stats/op_kernel/kernel_add_rms_norm_stats.cpp
)

# bw_probe: D3 Phase-0 benchmark-only probe (mode-0 memory pattern with the
# compute removed stage by stage). Own target, same reason as the norm kernel.
ascendc_library(no_workspace_kernel_bwprobe STATIC
    ${PROJECT_OP_SRC_BASE}/ops/bw_probe/op_kernel/kernel_bw_probe.cpp
)

# f3_floor: F3 (SwiGLU-in-gate_up-epilogue) Phase B ladder - catlass AIC/AIV mix
# kernel (paired gate/up blockMmad -> bf16 C workspace -> copy/silu epilogue).
# Own target, same reason as the others; catlass headers via the include dir
# below and CATLASS_ARCH defined source-locally (see the note underneath).
ascendc_library(no_workspace_kernel_f3 STATIC
    ${PROJECT_OP_SRC_BASE}/ops/f3_floor/op_kernel/kernel_f3_floor.cpp
)
ascendc_include_directories(no_workspace_kernel_f3 PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/catlass/include
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include
)

# catlass gates its 910B (arch 2201) block/epilogue templates on CATLASS_ARCH.
# The macro is defined source-locally in kernel_fa_fp32_stage1.cpp (NOT target-
# wide: a target-wide -D leaked into the lse_merge compile and broke its fp32
# path on device). The host-bisheng pass compiles the same source file and thus
# gets the define automatically; no forward-options hack needed either.
# ascendc_compile_definitions(no_workspace_kernel PRIVATE -DCATLASS_ARCH=2201)
# ascendc_compile_options(no_workspace_kernel PRIVATE -forward-options-to-host-compiler -DCATLASS_ARCH=2201)

# # kernel side files with workspace
# set(WORKSPACE_KERNEL_SRCS
#     ${PROJECT_OP_SRC_BASE}/ops/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp
# )
# if(BUILD_CATLASS_MODULE)
#     list(APPEND WORKSPACE_KERNEL_SRCS
#         ${PROJECT_OP_SRC_BASE}/ops/catlass/op_kernel/catlass_matmul_basic_kernel.cpp
#     )
# endif()
# ascendc_library(workspace_kernel STATIC ${WORKSPACE_KERNEL_SRCS})
# if(BUILD_CATLASS_MODULE)
#     ascendc_include_directories(workspace_kernel PRIVATE
#         ${CATLASS_DIR}/include
#     )
# endif()

# ascendc_compile_definitions(workspace_kernel PRIVATE
#        -DHAVE_WORKSPACE
#        -DHAVE_TILING
# )

# create shared library libascend_kernel.so
add_library(${OP_PLUGIN_NAME} SHARED ${OP_SRCS})

target_link_libraries(${OP_PLUGIN_NAME} PRIVATE
        # workspace_kernel
        no_workspace_kernel
        no_workspace_kernel_fa
        no_workspace_kernel_norm
        no_workspace_kernel_bwprobe
        no_workspace_kernel_f3
        torch_npu
        ascendcl
        tiling_api
        nnopbase
        opapi
        register
        platform
        ascendalog
        dl
)

target_link_directories(${OP_PLUGIN_NAME} PRIVATE
        ${TORCH_DIR}/lib
        ${TORCH_NPU_DIR}/lib
)

target_include_directories(${OP_PLUGIN_NAME} PRIVATE
        ${PROJECT_OP_SRC_BASE}/utils
        ${PROJECT_SOURCE_DIR}/include
        ${TORCH_DIR}/include
        ${TORCH_DIR}/include/torch/csrc/api/include
        ${TORCH_NPU_DIR}/include/third_party/acl/inc
        ${TORCH_NPU_DIR}/include/third_party/hccl/inc
        ${TORCH_NPU_DIR}/include
        ${PYTHON_INCLUDE_DIR}
        ${ASCEND_INCLUDE_DIR}/external
        ${ASCEND_INCLUDE_DIR}/experiment/platform
        ${ASCEND_INCLUDE_DIR}/experiment/runtime
)
