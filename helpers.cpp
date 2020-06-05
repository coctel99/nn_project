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
    cv::Mat resultImg(sizes[0], sizes[1], CV_8UC1);
    std::memcpy((void*)resultImg.data, tensor.data_ptr(), sizeof(at::ScalarType::Byte)*tensor.numel());

    return resultImg;
}

static at::Tensor matToTensor(cv::Mat &mat) {
    at::TensorOptions options(at::ScalarType::Byte);
    at::Tensor result = torch::from_blob(mat.data, {mat.rows, mat.cols}, options);
    //result = result.toType(torch::kUInt8);
    //result = result.toType(at::ScalarType::Float);

    return result;
}

static std::vector<std::vector<int>> matToArray(cv::Mat &mat) {
    std::vector<std::vector<int>> result;
    for (int i = 0; i < mat.rows; i++){
        result.insert(result.end(), mat.ptr<uchar>(i), mat.ptr<uchar>(i)+mat.cols);
    }

    return result;
}

static at::Tensor crop2fullmask(at::Tensor crop_mask, std::vector<int> bbox, cv::Size &im_size){
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

    //cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    //cv::imshow("Display Image", crop_mask_Mat);
    //cv::waitKey(0);

    cv::resize(crop_mask_Mat, crop_mask_Mat, {bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1}, cv::INTER_CUBIC);

    //std::vector<std::vector<int>> crop_mask_Arr = matToArray(crop_mask_Mat);

    std::cout << "crop_mask_Arr " << crop_mask.sizes() << std::endl;

    std::vector<int> inds = {22, 0, 141, 122};
    //std::vector<std::vector<int>> result(im_size.width, std::vector<int>(im_size.height));
    at::Tensor result = torch::zeros(torch::IntArrayRef {im_size.width, im_size.height});

    //TODO fix following part
    for(int i = 0; i < 2; i++) {
        result[bbox_valid[i + 1] + 1][bbox_valid[i] + 1] = crop_mask[inds[i + 1] + 1][inds[i] + 1];
    }

    return result;
}

static cv::Mat crop_from_bbox(cv::Mat &img, std::vector<int> &bbox, bool zero_pad=false){
    // Borders of image
    std::vector<int> bounds = {0, 0, img.rows - 1, img.cols - 1};
    std::cout << "crop_from_bbox bounds " << bounds << std::endl;

    // Valid bounding box locations as (x_min, y_min, x_max, y_max)
    std::vector<int> bbox_valid = {std::max(bbox[0], bounds[0]),
                                   std::max(bbox[1], bounds[1]),
                                   std::min(bbox[2], bounds[2]),
                                   std::min(bbox[3], bounds[3])};

    std::cout << "bbox_valid " << bbox_valid << std::endl;
    cv::Mat crop;
    at::Tensor img_tensor = matToTensor(img);
    std::cout << "img_tensor sizes " << img_tensor.sizes() << ", options " << img_tensor.options() << std::endl;

    cv::Mat img_tensor_mat = tensorToMat(img_tensor);
    if (!img_tensor_mat.data){
        std::cout << "No image data" << std::endl;
    }

    std::cout << "img_tensor_mat opened " << img_tensor_mat.size << ", options " << img_tensor_mat.channels() << std::endl;

    //cv::namedWindow("Display img_tensor_mat", cv::WINDOW_AUTOSIZE );
    //cv::imshow("Display img_tensor_mat", img_tensor_mat);
    //cv::waitKey(0);

    // if zero_pad
    // Initialize crop size (first 2 dimensions)
    at::TensorOptions options(at::ScalarType::Byte);
    at::Tensor crop_tensor = torch::zeros(torch::IntArrayRef {bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1}, options);
    //crop_tensor.toType(at::ScalarType::Byte);
    //crop_tensor = torch::transpose(crop_tensor, 0 ,1);

    // Offsets for x and y
    std::vector<int> offsets = {-bbox[0], -bbox[1]};
    std::cout << "offsets " << offsets << std::endl;

    std::cout << "crop_tensor sizes " << crop_tensor.sizes() << ", options " << crop_tensor.options() << std::endl;

    cv::Mat crop_tensor_mat = tensorToMat(crop_tensor);
    cv::namedWindow("Display crop_tensor_mat", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display crop_tensor_mat", crop_tensor_mat);
    cv::waitKey(0);

    //TODO add inds calculation
    std::vector<int> inds = {22, 0, 141, 122};
    // bbox_valid 0 137 119 259

    std::cout << "crop_from_bbox crop_tensor sizes " << crop_tensor.sizes() << std::endl;
    std::cout << "crop_from_bbox img_tensor sizes " << img_tensor.sizes() << std::endl;

    // Code from Python:
    //crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1] = img[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1]

    //crop_tensor.squeeze();
    crop_tensor = crop_tensor.index_put_(
            {torch::indexing::Slice{inds[1], inds[3]},
             torch::indexing::Slice{inds[0], inds[2]}},
            img_tensor.index(
             {torch::indexing::Slice{bbox_valid[1], bbox_valid[3]},
                     torch::indexing::Slice{bbox_valid[0], bbox_valid[2]}}));

    std::cout << "tensorToMat..."<< std::endl;
    crop = tensorToMat(crop_tensor);

    std::cout << "crop opened " << crop.size << ", channels " << crop.channels() << std::endl;
    cv::namedWindow("Display crop", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display crop", crop);
    cv::waitKey(0);

    return crop;
}

static cv::Mat make_masks_image(std::vector<std::vector<int>> &results) {
    //cv::Mat mask_image(width, height, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat mask_image = vecToMat(results);
    std::cout << "mask_image size: " << mask_image.size << std::endl;

    return mask_image;
}

static std::vector<std::vector<float>> make_gaussian(const std::vector<int> &size, const std::vector<int> &center, int sigma=10){
    std::vector<std::vector<int>> gaussian;
    std::vector<float> x;
    std::vector<float> y;
    float x0;
    float y0;
    x.reserve(size[1]);
    y.reserve(size[0]);
    for(int i = 0; i < size[1]; i++) {
        x.push_back(i);
    }
    for(int i = 0; i < size[0]; i++) {
        y.push_back(i);
    }

    if(center.empty()){
        x0 = y0 = size[0]/2;
    } else {
        y0 = center[0];
        x0 = center[1];
    }

    std::cout << "gauss x0, y0 " << x0 << " " << y0 << std::endl;

    std::vector<float> linear_x_x0 = x;
    for(int i = 0; i < size[1]; i++) {
        linear_x_x0[i] = pow(linear_x_x0[i] - x0, 2);
    }
    std::vector<float> linear_y_y0 = y;
    for(int i = 0; i < size[0]; i++) {
        linear_y_y0[i] = pow(linear_y_y0[i] - y0, 2);
    }

    std::vector<std::vector<float>> gauss(size[1], std::vector<float>(size[0], 0));
    for(int i = 0; i < size[1]; i++) {
        for(int j = 0; j < size[0]; j++) {
            gauss[i][j] = exp((-4 * log(2) * (linear_x_x0[i] + linear_y_y0[j])/pow(sigma, 2)));

            if (i == 0 && j == 0 || i == 50 && j == 50 || i == 180 && j == 283 || i == 511 && j == 511){
                std::cout << "gauss" << i << j << " " << gauss[i][j] << std::endl;
            }
        }
    }

    return gauss;
};

static std::vector<std::vector<float>> make_gt(const cv::Mat &img, const std::vector<std::vector<int>> &lables, int sigma=10){
    std::vector<std::vector<float>> gt;
    int h = img.rows;
    int w = img.cols;
    std::cout << "make_gt: height " << h << ", width " << w << std::endl;
    std::cout << "make_gt: lables size " << lables.size() << std::endl;
    for (int i = 0; i < lables.size(); i++) {
        std::cout << "lables[" << i << "]" << lables[i][0] << ", " << lables[i][1] << std::endl;
    }

    if(lables.empty()){
        std::cout << "make_gt: lables.empty()" << std::endl;
        gt = make_gaussian({h, w}, {h / 2, w / 2}, sigma=sigma);
    } else {
        if (lables.size() == 1) {
            // labels = labels[np.newaxis]
        }
        //gt.emplace_back(torch::zeros({h, w}));
        for (const auto & lable : lables) {
            gt = MAX(gt, make_gaussian({h, w}, {lable[0], lable[1]}, sigma));
            std::cout << "make_gt: make_gaussian with size " << h << ", " << w << " and center in " << lable[0] << ", " << lable[1] << std::endl;

        }
    }

    return gt;
}
