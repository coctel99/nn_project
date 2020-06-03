#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <string>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include "helpers.cpp"

static int find_dextr_bit_mask(const std::string &image_path, std::vector<std::vector<int>> extreme_points_double_array){
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
    //std::vector<std::vector<int>> v = {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
    //image = vecToMat(v);
    //image = cv::imread(image_path, cv::ImreadModes::IMREAD_COLOR);
    image = cv::imread(image_path, cv::ImreadModes::IMREAD_GRAYSCALE);
    cv::Size im_size = {image.rows, image.cols};
    if (!image.data){
        std::cout << "No image data" << std::endl;
        return -1;
    }
    cv::Mat bgr[3];
    cv::split(image, bgr);

    std::cout << "Image opened " << image.size << ", channels " << image.channels() << std::endl;

    // Crop image to the bounding box from the extreme points and resize
    std::vector<int> bbox = {-22, 137, 119, 259};
    cv::Mat crop_image = crop_from_bbox(image, bbox, true);
    cv::Mat resize_image;

    cv::resize(image, resize_image, cv::Size(512, 512));

    std::cout << "Image resized " << resize_image.size <<  ", channels " << resize_image.channels() << std::endl;

    // Generate extreme point heat map normalized to image values
    int minVal0 = INT_MAX;
    int minVal1 = INT_MAX;
    int PAD = 50;
    for(auto & i : extreme_points_double_array) {
        if (i[0] < minVal0){
            minVal0 = i[0];
        }
        if (i[1] < minVal1){
            minVal1 = i[1];
        }
    }

    std::cout << "cropsize " << crop_image.size() <<  ", channels " << crop_image.channels() << std::endl;


    std::vector<std::vector<int>> extreme_points = extreme_points_double_array;
    for(int i = 0; i < extreme_points.size(); i++){
        std::cout << "extreme_points[" << i << "]" << extreme_points[i][0] << ", " << extreme_points[i][1] << std::endl;
        //extreme_points[i][0] = 512 * (extreme_points[i][0] - minVal0 + PAD) * [1 / crop_image.size(), 1 / crop_image.size()[0]];
        //extreme_points[i][1] = extreme_points[i][1] - minVal1 + PAD;
        // (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]])
        std::cout << "extreme_points[" << i << "]" << extreme_points[i][0] << ", " << extreme_points[i][1] << std::endl;
        std::cout << std::endl;
    }

    std::cout << "Extreme points ok" << std::endl;

    std::vector<std::vector<float>> extreme_heatmap;
    extreme_heatmap = make_gt(resize_image, extreme_points, 10);
    //extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)

    std::cout << "Extreme heatmap ok " << extreme_heatmap.size() << std::endl;

    // Open image as tensor.
    //at::IntArrayRef sizes = {1, 4, 512, 512};
    at::Tensor tensor_image = matToTensor(resize_image);

    //std::vector<int64_t> sizes = {1, 4, 512, 512};
    //at::TensorOptions options(at::ScalarType::Byte);
    //at::Tensor tensor_image = torch::from_blob(image.data, at::IntList(sizes), options);
    //tensor_image = tensor_image.toType(at::kFloat);

    std::cout << "Image to tensor " << tensor_image.sizes() << std::endl;

    //Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    //inputs.push_back(torch::ones({1, 4, 512, 512}).to(device));
    inputs.emplace_back(tensor_image.to(device));

    std::cout << "Tensor to gpu" << std::endl;

    // Execute the model and turn its output into a tensor.
    std::cout << "Performing forward pass" << std::endl;
    at::Tensor outputs = module.forward(inputs).toTensor();
    outputs = torch::upsample_bilinear2d(outputs, {512, 512}, true);
    outputs = outputs.to(device=c10::DeviceType::CPU);

    std::cout << "Getting predictions" << std::endl;

    at::Tensor pred = outputs.transpose(1, 2);
    pred = 1 / (1 + exp(-pred));
    pred = at::squeeze(pred);

    cv::namedWindow("pred image", cv::WINDOW_AUTOSIZE );
    cv::imshow("pred image", tensorToMat(pred));
    cv::waitKey(0);

    std::cout << "pred sizes" << pred.sizes() << std::endl;

    at::Tensor bit_mask_array = crop2fullmask(pred, bbox, im_size);

    std::cout << "bit_mask_array sizes " << bit_mask_array.sizes() << std::endl;

    cv::Mat bit_mask_image = tensorToMat(bit_mask_array);
    cv::imwrite("../images/dextr-results/output.jpg", bit_mask_image);
    //cv::namedWindow("Display Image 3", cv::WINDOW_AUTOSIZE );
    //cv::imshow("Display Image 3", mask_image);
    //cv::waitKey(0);

    /* TODO
    image_with_bit_mask = add_mask_to_the_image(image_file, bit_mask_image)
     */
    return 0;
}