#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <string>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

static int find_dextr_bit_mask(std::string image_path, std::vector<std::vector<int>> extreme_points_double_array){
    //std::string model_file = "../resnet18.pt";
    std::string model_file = "../traced_model_gpu.pt";
    //std::string model_file = "../traced_model_cpu.pt";
    torch::DeviceType device = at::kCUDA;

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        std::cout << "Trying to open model " << model_file << std::endl;
        module = torch::jit::load(model_file);
        std::cout << "Success" << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }

    module.to(device);
    std::cout << "model moved to device" << std::endl;

    // Open the image file.
    // std::string image_path = "../images/dog-cat.jpg";
    cv::Mat image;
    image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (!image.data){
        printf("No image data \n");
        return -1;
    }

    // Preview image.
    // cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    // cv::imshow("Display Image", image);
    // cv::waitKey(0);

    // Resize image to 512x512.
    cv::Size size(512, 512);
    cv::Mat image_resized;
    cv::resize(image, image_resized, size);

    // Preview resized image.
    // cv::namedWindow("Display Image 2", cv::WINDOW_AUTOSIZE );
    // cv::imshow("Display Image 2", image);
    // cv::waitKey(0);

    // Open image as tensor.
    std::vector<int64_t> sizes = {1, 4, 512, 512};
    at::TensorOptions options(at::ScalarType::Byte);
    at::Tensor tensor_image = torch::from_blob(image.data, at::IntList(sizes), options);
    tensor_image = tensor_image.toType(at::kFloat);

    //Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    //inputs.push_back(torch::ones({1, 4, 512, 512}).to(device));
    inputs.push_back(tensor_image.to(device));

    // Execute the model and turn its output into a tensor.
    std::cout << "Performing forward pass" << std::endl;
    at::Tensor outputs = module.forward(inputs).toTensor();
    std::cout << outputs.slice(/*dim=*/1, /*start=*/0, /*end=*/1) << '\n';
    outputs = torch::upsample_bilinear2d(outputs, {512, 512}, true);
    outputs = outputs.to(device=c10::DeviceType::CPU);

    return 0;
}