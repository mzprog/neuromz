# NeuroMZ — an educational neural-network library in C11 (libc + libm only).
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build                    library, tests, examples
#   cmake --build build --target test      build, then run the tests (= ctest)
#   cmake --build build --target sanitize  ASan + UBSan build in build-san, then its tests (GCC/Clang)
#   cmake --build build --target bench     matrix-product and convolution benchmarks (slow)
#   cmake --build build --target data      download MNIST into data/, checksum-verified
#   cmake --build build --target data-text download the texts (Alice; Kalila wa-Dimna) into data/
#   cmake --install build --prefix DIR     headers, library and a CMake package: find_package(neuromz)
#
# Options: NMZ_OPENMP (multi-threaded matmul/attention, off), NMZ_CUDA (the CUDA
# backend, off; NMZ_CUDA_ARCHITECTURES = GPUs to compile for, default the local
# one), NMZ_WERROR (on when top-level), NMZ_SANITIZE (ASan + UBSan, off),
# NMZ_BUILD_TESTS / NMZ_BUILD_EXAMPLES (on when this is the top-level project).
#
# CMake 3.21 is the minimum: it is the first version that passes C11/C17
# to MSVC (/std:c11) — DESIGN decision 114.
cmake_minimum_required(VERSION 3.21)
project(neuromz VERSION 2.1.0 DESCRIPTION "Educational neural-network engine" LANGUAGES C)

set(NMZ_TOP ${PROJECT_IS_TOP_LEVEL})
option(NMZ_OPENMP "Multi-threaded matrix products and attention heads (OpenMP)" OFF)
option(NMZ_CUDA "CUDA backend: train on an NVIDIA GPU (needs the CUDA Toolkit; the library still runs without a GPU)" OFF)
set(NMZ_CUDA_ARCHITECTURES "" CACHE STRING
    "GPU architectures for NMZ_CUDA, e.g. 75;86;89 (each: machine code + PTX); empty = the local GPU (else 75)")
option(NMZ_WERROR "Treat compiler warnings as errors (default: only when this is the top-level project)" ${NMZ_TOP})
option(NMZ_SANITIZE "Build with AddressSanitizer + UndefinedBehaviorSanitizer (GCC/Clang)" OFF)
option(NMZ_BUILD_TESTS "Build the test suite" ${NMZ_TOP})
option(NMZ_BUILD_EXAMPLES "Build the C examples and benchmarks" ${NMZ_TOP})

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ------------------------------------------------------------------ warnings
# One interface target carries the warning set to everything built here (not
# to projects that use the library). GCC/Clang: the set of the old Makefile
# plus -Wvla (MSVC has no VLAs). MSVC: /W4 minus the implicit-narrowing
# warnings C4244/C4267/C4305, which GCC and Clang only give with -Wconversion
# (not enabled there either) — decision 115.
# (C sources only: the CUDA sources get nvcc's own set below.)
add_library(nmz_warnings INTERFACE)
set(NMZ_C "$<COMPILE_LANGUAGE:C>")
if(MSVC)
  target_compile_options(nmz_warnings INTERFACE $<${NMZ_C}:/W4 /wd4244 /wd4267 /wd4305 /utf-8 $<$<BOOL:${NMZ_WERROR}>:/WX>>)
  target_compile_definitions(nmz_warnings INTERFACE _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(nmz_warnings INTERFACE $<${NMZ_C}:-Wall -Wextra -pedantic -Wshadow -Wstrict-prototypes
                                                -Wmissing-prototypes -Wvla $<$<BOOL:${NMZ_WERROR}>:-Werror>>)
  # no fused multiply-add contraction: a·b + c rounds twice everywhere, so
  # the blocked matmul stays bit-identical to the naive loops (and the KV
  # cache to recomputation) on CPUs with FMA too. GCC already does this in
  # ISO C mode; Clang (Apple's in particular, on arm64) contracts by
  # default — decision 120. MSVC does not contract under /fp:precise.
  target_compile_options(nmz_warnings INTERFACE $<${NMZ_C}:-ffp-contract=off>)
endif()
if(NMZ_SANITIZE)
  if(MSVC)
    message(FATAL_ERROR "NMZ_SANITIZE is for GCC/Clang (MSVC's /fsanitize=address has no UBSan)")
  endif()
  set(NMZ_SAN_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  target_compile_options(nmz_warnings INTERFACE $<${NMZ_C}:${NMZ_SAN_FLAGS}>)
  target_link_options(nmz_warnings INTERFACE ${NMZ_SAN_FLAGS})
endif()

# ------------------------------------------------------------------- library
set(NMZ_LIB_SOURCES
    src/error.c src/tensor.c src/rng.c src/activation.c src/dense.c src/loss.c src/net.c src/train.c
    src/trace.c src/gradcheck.c src/io.c src/dataset.c src/defaults.c src/conv2d.c src/shape_layers.c
    src/normalize.c src/text.c src/recurrent.c src/lstm.c src/gru.c src/generate.c src/matmul.c
    src/attention.c src/transformer.c src/backend.c src/backend_cpu.c)
add_library(neuromz STATIC ${NMZ_LIB_SOURCES})
add_library(neuromz::neuromz ALIAS neuromz)
set_target_properties(neuromz PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
                                         POSITION_INDEPENDENT_CODE ON)
target_include_directories(neuromz PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                          $<INSTALL_INTERFACE:include>)
target_link_libraries(neuromz PRIVATE $<BUILD_INTERFACE:nmz_warnings>)
if(NOT WIN32)
  target_link_libraries(neuromz PUBLIC m) # libm (part of the C runtime on Windows)
endif()
# ---------------------------------------------------------------- CUDA backend
# The kernels are compiled by nvcc behind a C ABI (src/cuda/backend_cuda.cu,
# one function seen by the C sources); the public headers stay pure C. The
# CUDA runtime is linked statically (it finds the driver at run time and
# reports a missing one); cuBLAS is loaded at run time, so a program built
# with NMZ_CUDA starts on any machine and falls back to the CPU (decision 140).
set(NMZ_CUDA_LINKED 0)
if(NMZ_CUDA)
  set(_nmz_arch "${NMZ_CUDA_ARCHITECTURES}")
  if(NOT _nmz_arch)
    find_program(NMZ_NVIDIA_SMI nvidia-smi)
    if(NMZ_NVIDIA_SMI)
      execute_process(COMMAND ${NMZ_NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
                      OUTPUT_VARIABLE _nmz_cc RESULT_VARIABLE _nmz_r OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
      if(_nmz_r EQUAL 0 AND _nmz_cc MATCHES "^([0-9]+)\\.([0-9]+)")
        set(_nmz_arch "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
      endif()
    endif()
    if(NOT _nmz_arch)
      set(_nmz_arch 75) # Turing: the oldest architecture CUDA 13 compiles for
    endif()
    message(STATUS "neuromz: CUDA architectures ${_nmz_arch} (the local GPU, or 75; set NMZ_CUDA_ARCHITECTURES for others)")
  endif()
  # every entry without -real/-virtual = machine code + PTX (the PTX of the
  # newest one lets later GPUs run the library through the driver's JIT)
  set(_nmz_min 9999)
  foreach(_a IN LISTS _nmz_arch)
    if(_a MATCHES "^([0-9]+)" AND CMAKE_MATCH_1 LESS _nmz_min)
      set(_nmz_min ${CMAKE_MATCH_1})
    endif()
  endforeach()
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES ${_nmz_arch})
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(neuromz PRIVATE src/cuda/backend_cuda.cu)
  set_target_properties(neuromz PROPERTIES CUDA_ARCHITECTURES "${_nmz_arch}" CUDA_STANDARD 17
                                           CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Static)
  target_compile_definitions(neuromz PRIVATE NMZ_HAVE_CUDA=1 NMZ_CUDA_MIN_CC=${_nmz_min})
  # --fmad=false: no fused multiply-add in the kernels, as on the CPU (decision 120): the optimizers then
  # round exactly like the CPU's and the element-wise ops differ from it only where the math library does
  target_compile_options(neuromz PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false $<$<BOOL:${NMZ_WERROR}>:-Werror=all-warnings>>)
  target_link_libraries(neuromz PUBLIC CUDA::cudart_static)
  if(NOT WIN32)
    target_link_libraries(neuromz PUBLIC ${CMAKE_DL_LIBS}) # dlopen for cuBLAS
  endif()
  set(NMZ_CUDA_LINKED 1)
endif()

set(NMZ_OPENMP_LINKED 0)
if(NMZ_OPENMP)
  find_package(OpenMP COMPONENTS C)
  if(OpenMP_C_FOUND)
    target_link_libraries(neuromz PUBLIC OpenMP::OpenMP_C)
    set(NMZ_OPENMP_LINKED 1)
  else()
    message(WARNING "NMZ_OPENMP=ON but this compiler has no OpenMP: building single-threaded")
  endif()
endif()

# --------------------------------------------------------------------- tests
if(NMZ_BUILD_TESTS)
  enable_testing()
  add_executable(test_neuromz
      tests/test_main.c tests/test_core.c tests/test_math.c tests/test_train.c tests/test_io.c
      tests/test_layers.c tests/test_optim.c tests/test_text.c tests/test_seq.c tests/test_gated.c
      tests/test_transformer.c tests/test_backend.c)
  set_target_properties(test_neuromz PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
  target_include_directories(test_neuromz PRIVATE src)
  target_link_libraries(test_neuromz PRIVATE neuromz nmz_warnings)

  # `--target test` (and plain ctest) first brings the build up to date
  add_test(NAME build COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} "$<$<BOOL:$<CONFIG>>:--config;$<CONFIG>>"
           COMMAND_EXPAND_LISTS)
  set_tests_properties(build PROPERTIES FIXTURES_SETUP built)
  # fixtures and example data are read relative to the source tree; scratch
  # files go to the build tree
  add_test(NAME unit COMMAND test_neuromz WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  set_tests_properties(unit PROPERTIES FIXTURES_REQUIRED built ENVIRONMENT "NMZ_TEST_DIR=${CMAKE_BINARY_DIR}")
  if(NMZ_CUDA)
    # the GPU tests (label gpu): every op against the CPU, gradient checks on
    # the GPU, training parity; exit 77 (skipped) without a usable GPU
    add_executable(test_gpu tests/test_gpu.c tests/test_gpu_main.c)
    set_target_properties(test_gpu PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
    target_include_directories(test_gpu PRIVATE src)
    target_link_libraries(test_gpu PRIVATE neuromz nmz_warnings)
    add_test(NAME gpu COMMAND test_gpu WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    add_test(NAME gpu_parity COMMAND test_gpu --parity WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    set_tests_properties(gpu gpu_parity PROPERTIES LABELS gpu SKIP_RETURN_CODE 77 FIXTURES_REQUIRED built
                                                   ENVIRONMENT "NMZ_TEST_DIR=${CMAKE_BINARY_DIR}")
    # the same build on a machine "without a GPU": the CPU suite still passes
    # and auto falls back to the CPU
    add_test(NAME unit_no_gpu COMMAND test_neuromz backend WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    set_tests_properties(unit_no_gpu PROPERTIES FIXTURES_REQUIRED built
                         ENVIRONMENT "NMZ_TEST_DIR=${CMAKE_BINARY_DIR};CUDA_VISIBLE_DEVICES=-1")
  endif()
  if(NMZ_SANITIZE)
    set_property(TEST unit APPEND PROPERTY ENVIRONMENT "ASAN_OPTIONS=detect_leaks=1"
                                                       "UBSAN_OPTIONS=print_stacktrace=1")
  endif()
endif()

# ---------------------------------------------------------- examples, bench
if(NMZ_BUILD_EXAMPLES)
  foreach(ex api_xor api_mnist bench_conv bench_matmul bench_device)
    add_executable(${ex} examples/${ex}.c)
    set_target_properties(${ex} PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
    target_link_libraries(${ex} PRIVATE neuromz nmz_warnings)
  endforeach()
  if(NMZ_BUILD_TESTS)
    add_test(NAME example_xor COMMAND api_xor WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    # MNIST regression: only when `--target data` has been run (exit 77 = skipped)
    add_test(NAME example_mnist COMMAND api_mnist data WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    set_tests_properties(example_xor example_mnist PROPERTIES FIXTURES_REQUIRED built)
    set_tests_properties(example_mnist PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  add_custom_target(bench
      COMMAND bench_matmul 5
      COMMAND bench_conv
      DEPENDS bench_matmul bench_conv
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      USES_TERMINAL
      COMMENT "Benchmarks of the C API (slow)")
endif()

# ------------------------------------------------------------ data, sanitize
# Downloads go to data/ in the source tree (gitignored); sh on Unix-like
# systems, PowerShell on Windows — both verify the same checksums.
if(WIN32)
  find_program(NMZ_POWERSHELL NAMES pwsh powershell) # PowerShell 7 if installed, else Windows PowerShell 5.1
  set(NMZ_DATA_MNIST ${NMZ_POWERSHELL} -NoProfile -ExecutionPolicy Bypass -File scripts/get_mnist.ps1 data)
  set(NMZ_DATA_TEXT ${NMZ_POWERSHELL} -NoProfile -ExecutionPolicy Bypass -File scripts/get_text.ps1 data)
else()
  set(NMZ_DATA_MNIST sh scripts/get_mnist.sh data)
  set(NMZ_DATA_TEXT sh scripts/get_text.sh data)
endif()
if(NMZ_TOP)
  add_custom_target(data COMMAND ${NMZ_DATA_MNIST} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} USES_TERMINAL
                    COMMENT "Downloading MNIST into data/")
  add_custom_target(data-text COMMAND ${NMZ_DATA_TEXT} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} USES_TERMINAL
                    COMMENT "Downloading the texts into data/")
  if(NOT MSVC AND NOT NMZ_SANITIZE)
    # a second build tree with ASan + UBSan, then its tests
    add_custom_target(sanitize
        COMMAND ${CMAKE_COMMAND} -S ${PROJECT_SOURCE_DIR} -B ${PROJECT_SOURCE_DIR}/build-san
                -DCMAKE_BUILD_TYPE=Debug -DNMZ_SANITIZE=ON -DNMZ_OPENMP=${NMZ_OPENMP}
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        COMMAND ${CMAKE_COMMAND} --build ${PROJECT_SOURCE_DIR}/build-san
        COMMAND ${CMAKE_CTEST_COMMAND} --test-dir ${PROJECT_SOURCE_DIR}/build-san --output-on-failure
        USES_TERMINAL
        COMMENT "ASan + UBSan build and tests in build-san/")
  endif()
endif()

# ------------------------------------------------------------------- install
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS neuromz EXPORT neuromzTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES include/neuromz.h include/neuromz_layer.h include/neuromz_backend.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT neuromzTargets NAMESPACE neuromz:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/neuromz)
configure_package_config_file(cmake/neuromzConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/neuromzConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/neuromz)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/neuromzConfigVersion.cmake
                                 COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/neuromzConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/neuromzConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/neuromz)
