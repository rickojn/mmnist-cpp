cmake_minimum_required(VERSION 3.10)

# Define the project name and languages
project(mlp_project CXX)
# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# Set compile flags for Debug and Release configurations
set(CMAKE_CXX_FLAGS_DEBUG "-g -fopenmp -Wall -O2 -mavx -mavx2 -mfma")
set(CMAKE_CXX_FLAGS_RELEASE "-fopenmp -Wall -O2 -DNDEBUG -mavx -mavx2 -mfma -march=native")

add_subdirectory(libs/custard-flow)

# Define the source files
set(SOURCE_FILES
    src/mlp.cpp
)


# Specify the output binary
add_executable(mlp ${SOURCE_FILES})

# Link the math library (libm)
target_link_libraries(mlp m custard-flow)

# cd build/
# cmake -DCMAKE_BUILD_TYPE=Release ..
# cd ..
# cmake --build build/ --config Release
# ./build/mlp
