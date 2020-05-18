# Instruction
## Install
1. LibTorch C++
<br>_Note: better to download the debug version_
1. NVIDIA CUDA toolkit 10.1
<br>_Note: version 10.1 is the only compatible_
3. NVIDIA cuDNN

LibTorch works only with MSVC compiler, that is why you should have 
Visual Studio installed.

## Edit CMake file
Check the CMakeLists.txt to have the correct:
1. Project name
2. Executable file name
3. Torch_DIR
<br>_Note: must be an absolute path to libtorch/share/cmake/Torch_
4. CMAKE_PREFIX_PATH
<br>_Note: must be an absolute path to libtorch/share/cmake/Torch_
5. CUDNN_LIBRARY
<br>_Note: must be an absolute path to CUDA library files_
<br> _example: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/lib/x64_
6. CUDNN_INCLUDE_DIR
<br>_Note: must be an absolute path to CUDA library include files_
<br> _example: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/include_
## Open the traced model file
1. Download the traced_model_gpu.pt if you want to use GPU,
 or traced_model_cpu.pt if not
2. Put downloaded model file in the project directory
3. Main.cpp preforms opening model file and its deserialization

To check everything is OK, forward pass with input array of size 
{1, 4, 512, 512}, containing ones, is performed and one-dimesional 
slice of output is printed.