cmake_minimum_required(VERSION 3.16)
project(MPI_Test CXX)
set(CMAKE_CXX_STANDARD 20)
find_package(MPI REQUIRED)

add_executable(lat2d lat2d_mpi.cpp)
target_link_libraries(lat2d MPI::MPI_CXX)
target_compile_options(lat2d PRIVATE
        -O3            # optimisation forte
        -march=native  # utilise les instructions CPU locales
        -Wall -Wextra  # warnings utiles
        -Wpedantic
)
