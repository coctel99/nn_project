#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <algorithm>
#include <string>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include "helpers.cpp"

static int find_dextr_bit_mask(const std::string &image_path, const std::vector<std::vector<int>> &extreme_points_double_array){
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
    cv::Mat image = cv::imread(image_path, cv::ImreadModes::IMREAD_COLOR);
    //image = cv::imread(image_path, cv::ImreadModes::IMREAD_GRAYSCALE);
    cv::Size im_size = {image.rows, image.cols};
    if (!image.data){
        std::cout << "No image data" << std::endl;
        return -1;
    }
    cv::Mat bgr[3];
    cv::split(image, bgr);

    std::cout << "Image opened " << image.size << ", channels " << image.channels() << std::endl;

    // Crop image to the bounding box from the extreme points and resize
    // TODO: add bbox calculation
    std::vector<int> bbox = {-22, 137, 119, 259};
    cv::Mat crop_image = crop_from_bbox(image, bbox, true);
    cv::Mat resize_image;
    std::vector<std::vector<int>> extreme_points = extreme_points_double_array;

    cv::resize(image, resize_image, cv::Size(512, 512));
    std::cout << "Image resized " << resize_image.size <<  ", channels " << resize_image.channels() << std::endl;

    // Generate extreme point heat map normalized to image values
    int minVal0 = INT_MAX;
    int minVal1 = INT_MAX;
    int PAD = 50;
    for(auto & i : extreme_points) {
        if (i[0] < minVal0){
            minVal0 = i[0];
        }
        if (i[1] < minVal1){
            minVal1 = i[1];
        }
    }

    for(auto & extreme_point : extreme_points){
        extreme_point[0] = (int) 512 * (extreme_point[0] - minVal0 + PAD) * ((float) 1 / crop_image.size().width);
        extreme_point[1] = (int) 512 * (extreme_point[1] - minVal1 + PAD) * ((float) 1 / crop_image.size().height);
        //std::cout << "extreme_points[" << i << "] " << extreme_points[i][0] << ", " << extreme_points[i][1]<< std::endl;
        //std::cout << std::endl;
    }

    std::cout << "Extreme points ok" << std::endl;

    at::Tensor extreme_heatmap;
    extreme_heatmap = make_gt(resize_image, extreme_points, 10);
    //extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)

    std::cout << "Extreme heatmap ok " << extreme_heatmap.sizes() << std::endl;

    // Open image as tensor.
    //at::IntArrayRef sizes = {1, 4, 512, 512};
    at::Tensor tensor_image = matToTensor(resize_image).to(torch::kFloat32);
    std::cout << "tensor_image sizes " << tensor_image.sizes() << ", options " << tensor_image.options() << std::endl;
    std::cout << "extreme_heatmap sizes " << extreme_heatmap.sizes() << ", options " << extreme_heatmap.options() << std::endl;

    extreme_heatmap = extreme_heatmap.reshape({512, 512, 1});
    std::cout << "extreme_heatmap sizes " << extreme_heatmap.sizes() << ", options " << extreme_heatmap.options() << std::endl;

    at::Tensor input_dextr = torch::cat({tensor_image, extreme_heatmap}, 2);
    std::cout << "input_dextr sizes " << input_dextr.sizes() << ", options " << input_dextr.options() << std::endl;

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    //inputs.push_back(torch::ones({1, 4, 512, 512}).to(device));
    inputs.emplace_back(input_dextr.transpose(2, 0).unsqueeze(0).to(device));

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

    pred = 255 * pred;
    pred = pred.to(at::ScalarType::Byte);
    cv::Mat pred_mat = tensorToMat(pred);
    cv::namedWindow("pred image", cv::WINDOW_AUTOSIZE );
    cv::imshow("pred image", pred_mat);
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