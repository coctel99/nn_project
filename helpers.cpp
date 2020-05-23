#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <utility>


static cv::Mat tensorToMat(at::Tensor &tensor) {
    auto sizes = tensor.sizes();
    cv::Mat resultImg(sizes[0], sizes[1], 0);
    std::memcpy((void*)resultImg.data, tensor.data_ptr(), sizeof(torch::kU8)*tensor.numel());
    return resultImg;
}

static std::vector<std::vector<int>> matToArray(cv::Mat &mat) {
    std::vector<std::vector<int>> result;
    for (int i = 0; i < mat.rows; i++){
        result.insert(result.end(), mat.ptr<uchar>(i), mat.ptr<uchar>(i)+mat.cols);
    }
    return result;
}

static std::vector<std::vector<int>> crop2fullmask(at::Tensor crop_mask, std::vector<int> bbox, cv::Size &im_size){
    // Borders of image
    std::vector<int> bounds = {0, 0, im_size.height - 1, im_size.width - 1};

    // Valid bounding box locations as (x_min, y_min, x_max, y_max)
    std::vector<int> bbox_valid = {std::max(bbox[0], bounds[0]),
                                   std::max(bbox[1], bounds[1]),
                                   std::min(bbox[2], bounds[2]),
                                   std::min(bbox[3], bounds[3])};

    // Simple per element addition in the tuple
    std::vector<int> offsets = {-bbox_valid[0], -bbox_valid[1]};

    cv::Mat crop_mask_Mat = tensorToMat(crop_mask);

    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", crop_mask_Mat);
    cv::waitKey(0);

    cv::resize(crop_mask_Mat, crop_mask_Mat, {bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1}, cv::INTER_CUBIC);

    cv::namedWindow("Display Image 2", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image 2", crop_mask_Mat);
    cv::waitKey(0);

    return matToArray(crop_mask_Mat);
}

static cv::Mat make_masks_image(std::vector<std::vector<int>> &results) {
    int width, height = results.size();
    //cv::Mat mask_image(width, height, CV_8UC3, cv::Scalar(0, 0, 0));
    std::cout << "make_masks_image sizes: " << width << height << " || " << results.size() << std::endl;

    cv::Mat mask_image(512, 512, CV_8UC3, cv::Scalar(0, 0, 0));
    return mask_image;
}

