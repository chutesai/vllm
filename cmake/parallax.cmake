# Parallax's SM120 sparse ternary expert kernel. No worker-side compilation.
if(VLLM_TARGET_DEVICE STREQUAL "cuda" AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  find_library(PARALLAX_TORCH_PYTHON torch_python
    PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
  define_extension_target(
    _parallax_C
    DESTINATION vllm
    LANGUAGE CUDA
    SOURCES "csrc/parallax/sparse_ternary_fp4.cu"
    COMPILE_FLAGS "-O3;-gencode=arch=compute_120a,code=sm_120a;--expt-relaxed-constexpr"
    INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}"
    LIBRARIES "${PARALLAX_TORCH_PYTHON}"
    WITH_SOABI)
  set_target_properties(_parallax_C PROPERTIES CUDA_ARCHITECTURES OFF)
elseif(VLLM_PARALLAX_ONLY)
  message(FATAL_ERROR "Parallax sparse FP4 requires CUDA 13 or newer")
else()
  # The BF16 path works without this architecture-specific extension.
  add_custom_target(_parallax_C)
endif()
