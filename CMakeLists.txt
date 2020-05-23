cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(nn-app)

set(Torch_DIR "D:/nn_projects/Include/libtorch/share/cmake/Torch")
set(OpenCV_DIR "D:/nn_projects/Include/opencv/build")
set(CMAKE_PREFIX_PATH "D:/nn_projects/Include/libtorch/share/cmake/Torch")

set(CUDNN_LIBRARY "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/lib/x64")
set(CUDNN_INCLUDE_DIR"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/include")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(nn-app main.cpp dextr.cpp helpers.cpp)
target_link_libraries(nn-app "${TORCH_LIBRARIES}")
target_link_libraries(nn-app ${OpenCV_LIBS})
set_property(TARGET nn-app PROPERTY CXX_STANDARD 14)

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET nn-app
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:nn-app>)
endif (MSVC)