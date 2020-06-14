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

static  at::Tensor vecToTensor(const std::vector<std::vector<int>> &vector) {
    at::TensorOptions options(torch::kInt32);
    at::Tensor result = torch::from_blob((void *) vector.data(), {(int) vector.size(), (int) vector[0].size()}, options);
    return result;
}

static cv::Mat tensorToMat(at::Tensor &tensor) {
    auto sizes = tensor.sizes();
    cv::Mat result(sizes[0], sizes[1], CV_8UC1);
    std::memcpy((void*)result.data, tensor.data_ptr(), sizeof(at::ScalarType::Byte)*tensor.numel());

    return result;
}

static at::Tensor matToTensor(cv::Mat &mat) {
    at::TensorOptions options(at::ScalarType::Byte);
    at::Tensor result = torch::from_blob(mat.data, {mat.rows, mat.cols}, options);
    //result = result.toType(torch::kUInt8);
    //result = result.toType(at::ScalarType::Float);

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

    //TODO: add inds calculation
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

    //std::cout << "crop opened " << crop.size << ", channels " << crop.channels() << std::endl;
    //cv::namedWindow("Display crop", cv::WINDOW_AUTOSIZE );
    //cv::imshow("Display crop", crop);
    //cv::waitKey(0);

    return crop;
}

static cv::Mat make_masks_image(std::vector<std::vector<int>> &results) {
    //cv::Mat mask_image(width, height, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat mask_image = vecToMat(results);
    std::cout << "mask_image size: " << mask_image.size << std::endl;

    return mask_image;
}

static at::Tensor make_gaussian(const std::vector<int> &size, const std::vector<int> &center, int sigma=10){
    std::cout << "make_gaussian..." << std::endl;
    int x0;
    int y0;
    at::TensorOptions options(at::ScalarType::Float);
    std::cout << "arange sizes: " << size[0] << ", " << size[1] << std::endl;
    at::Tensor x = torch::arange((float) size[1], options);
    at::Tensor y = torch::arange( (float) size[0], options);
    std::cout << "x sizes: " << x.sizes() << std::endl;
    std::cout << "y sizes: " << y.sizes() << std::endl;
    y = torch::reshape(y, {size[0], 1});
    std::cout << "y sizes: " << y.sizes() << std::endl;

    if(center.empty()){
        x0 = size[0]/2;
        y0 = size[0]/2;
    } else {
        x0 = center[0];
        y0 = center[1];
    }

    std::cout << "gauss x0, y0: " << x0 << " " << y0 << std::endl;

    //at::Tensor gauss = torch::zeros(torch::IntArrayRef {size[0], size[1]}, options);
    at::Tensor gauss = torch::exp( -4 * log(2) * (pow((x - x0), 2) + pow((y - y0), 2)) / pow(sigma, 2));
    std::cout << "gauss sizes: " << gauss.sizes() << ", options: " << gauss.options() << std::endl;

    at::Tensor gauss_mat_tensor = (255 * gauss).to(at::ScalarType::Byte);
    cv::Mat gauss_mat = tensorToMat(gauss_mat_tensor);
    std::cout << "crop gauss_mat " << gauss_mat.size << ", channels " << gauss_mat.channels() << std::endl;
    cv::namedWindow("Display gauss_mat", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display gauss_mat", gauss_mat);
    cv::waitKey(0);

    return gauss;
};

static at::Tensor make_gt(const cv::Mat &img, const std::vector<std::vector<int>> &lables, int sigma=10){
    int h = img.rows;
    int w = img.cols;
    at::TensorOptions options(at::ScalarType::Float);
    at::Tensor gt = torch::zeros(torch::IntArrayRef {h, w}, options);

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
            std::cout << "torch::max..."<< std::endl;
            //gt = torch::max(gt, make_gaussian({h, w}, {lable[0], lable[1]}, sigma));
            gt = torch::max(gt, make_gaussian({h, w}, {lable[0], lable[1]}, sigma));
            std::cout << "make_gt: make_gaussian with size " << h << ", " << w << " and center in " << lable[0] << ", " << lable[1] << std::endl;

        }
    }

    std::cout << "tensorToMat..."<< std::endl;
    at::Tensor gauss_mat_tensor = (255 * gt).to(at::ScalarType::Byte);
    cv::Mat gt_mat = tensorToMat(gauss_mat_tensor);

    std::cout << "gt_mat opened " << gt_mat.size << ", channels " << gt_mat.channels() << std::endl;
    cv::namedWindow("Display gt_mat", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display gt_mat", gt_mat);
    cv::waitKey(0);

    return gt;
}
