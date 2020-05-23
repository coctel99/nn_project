#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <utility>

static  cv::Mat vecToMat(const std::vector<std::vector<int>> &vecIn) {
    cv::Mat matOut(vecIn.size(), vecIn[0].size(), CV_32S);
    for (int i = 0; i < matOut.rows; i++) {
        for (int j = 0; j < matOut.cols; j++) {
            matOut.at<int>(i, j) = 1./(i + j + 1);
        }
    }

    return matOut;
}

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

    std::cout << "bounds " << bounds << std::endl;

    // Valid bounding box locations as (x_min, y_min, x_max, y_max)
    std::vector<int> bbox_valid = {std::max(bbox[0], bounds[0]),
                                   std::max(bbox[1], bounds[1]),
                                   std::min(bbox[2], bounds[2]),
                                   std::min(bbox[3], bounds[3])};

    std::cout << "bbox_valid " << bbox_valid << std::endl;

    // Simple per element addition in the tuple
    std::vector<int> offsets = {-bbox[0], -bbox[1]};

    std::cout << "offsets " << offsets << std::endl;

    cv::Mat crop_mask_Mat = tensorToMat(crop_mask);

    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", crop_mask_Mat);
    cv::waitKey(0);

    cv::resize(crop_mask_Mat, crop_mask_Mat, {bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1}, cv::INTER_CUBIC);

    cv::namedWindow("Display Image 2", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image 2", crop_mask_Mat);
    cv::waitKey(0);

    std::vector<std::vector<int>> crop_mask_Arr = matToArray(crop_mask_Mat);

    std::cout << "crop_mask_Arr " << crop_mask_Arr.size() << std::endl;

    std::vector<int> inds = {22, 0, 141, 122};
    std::vector<std::vector<int>> result(im_size.width, std::vector<int>(im_size.height));
    for(int i = 0; i < 2; i++) {
        result[bbox_valid[i + 1] + 1, bbox_valid[i] + 1] = crop_mask_Arr[inds[i + 1] + 1, inds[i] + 1];
    }

    return result;
}

static cv::Mat make_masks_image(std::vector<std::vector<int>> &results) {
    //cv::Mat mask_image(width, height, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat mask_image = vecToMat(results);
    std::cout << "mask_image size: " << mask_image.size << std::endl;

    return mask_image;
}

