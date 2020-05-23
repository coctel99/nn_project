#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <string>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include "helpers.cpp"

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
    cv::Mat image;
    image = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Size im_size = {image.rows, image.cols};
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
    outputs = torch::upsample_bilinear2d(outputs, {512, 512}, true);
    outputs = outputs.to(device=c10::DeviceType::CPU);

    at::Tensor pred = outputs.transpose(1, 2);
    pred = 1 / (1 + exp(-pred));
    pred = at::squeeze(pred);

    std::cout << "pred sizes" << pred.sizes() << std::endl;

    std::vector<int> bbox = {-22, 137, 119, 259};

    std::vector<std::vector<int>> bit_mask_array = crop2fullmask(pred, bbox, im_size);

    std::cout << "bit_mask_array sizes " << bit_mask_array.size() << " " << bit_mask_array[0].size() << std::endl;

    cv::Mat bit_mask_image = make_masks_image(bit_mask_array);

    //cv::namedWindow("Display Image 3", cv::WINDOW_AUTOSIZE );
    //cv::imshow("Display Image 3", mask_image);
    //cv::waitKey(0);

    /* TODO
    image_with_bit_mask = add_mask_to_the_image(image_file, bit_mask_image)
     */
    return 0;
}