cmake_minimum_required(VERSION 3.13)
project(ford_fulkerson_gpu)

set(CMAKE_CXX_STANDARD 14)

add_executable(ford_fulkerson_gpu serial_ford_fulkerson.cpp graph_generator.cpp graph_generator.h)