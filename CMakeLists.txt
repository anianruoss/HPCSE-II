cmake_minimum_required(VERSION 3.10)

project(HPCSE-II)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -O3 -march=native")

add_subdirectory(exercise01)
