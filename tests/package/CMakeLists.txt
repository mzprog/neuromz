# tests/package — a minimal consumer of an installed NeuroMZ (CI): checks
# that find_package(neuromz) works and the library links and runs.
cmake_minimum_required(VERSION 3.21)
project(use_neuromz LANGUAGES C)
find_package(neuromz 2.1 REQUIRED)
add_executable(use_neuromz use_neuromz.c)
target_link_libraries(use_neuromz PRIVATE neuromz::neuromz)
