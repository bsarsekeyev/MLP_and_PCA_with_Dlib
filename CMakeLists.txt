cmake_minimum_required(VERSION 3.10.0)
project(mlp-dlib-iris)

# choose the folder where DLIB is built 
set(DLIB_PATH PATH "$ENV{HOME}/development") 

set(CMAKE_VERBOSE_MAKEFILE ON)

set(requiredlibs "cblas")
list(APPEND requiredlibs "lapack")
list(APPEND requiredlibs "stdc++")
list(APPEND requiredlibs "stdc++fs")

find_package(CUDA)
if (CUDA_FOUND)
  list(APPEND requiredlibs ${CUDA_LIBRARIES})
  list(APPEND requiredlibs ${CUDA_CUBLAS_LIBRARIES})
  list(APPEND requiredlibs ${CUDA_cusolver_LIBRARY})
  list(APPEND requiredlibs ${CUDA_curand_LIBRARY})

  get_filename_component(cudnn_hint_path "${CUDA_CUBLAS_LIBRARIES}" PATH)
  find_library(cudnn cudnn
      HINTS ${cudnn_hint_path} ENV CUDNN_LIBRARY_DIR  ENV CUDNN_HOME
      PATHS /usr/local /usr/local/cuda ENV LD_LIBRARY_PATH
      PATH_SUFFIXES lib64 lib x64
      )
  message("CUDNN found ${cudnn}")
  list(APPEND requiredlibs ${cudnn})
endif()

list(APPEND requiredlibs X11)
list(APPEND requiredlibs png jpeg)

set(CMAKE_CXX_FLAGS "-std=c++17 -msse3 -fopenmp -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

include_directories(${DLIB_PATH}/libs/include)
link_directories(${DLIB_PATH}/libs/lib)
link_directories(${DLIB_PATH}/libs/lib64)

set(SOURCES
    main.cc
    )

add_executable(mlp-dlib ${SOURCES})
target_link_libraries(mlp-dlib optimized dlib debug dlib)
target_link_libraries(mlp-dlib  ${requiredlibs})