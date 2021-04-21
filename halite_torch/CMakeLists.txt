cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(halite_torch)

find_package(Torch REQUIRED)

add_executable(halite_torch halite_torch.cpp)
target_link_libraries(halite_torch "${TORCH_LIBRARIES}")
set_property(TARGET halite_torch PROPERTY CXX_STANDARD 14)
