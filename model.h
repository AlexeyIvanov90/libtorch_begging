#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "data_set.h"


struct ConvNetImpl : public torch::nn::Module 
{
	ConvNetImpl();
	ConvNetImpl(int64_t channels, int64_t height, int64_t width);
	torch::Tensor forward(torch::Tensor x);
	int64_t get_conv_output(int64_t channels, int64_t height, int64_t width);

	torch::nn::Conv2d conv1, conv2;
	int64_t n;
	torch::nn::Linear lin1;
};


TORCH_MODULE(ConvNet);


torch::Tensor classification(torch::Tensor img_tensor, ConvNet model);
void classification_data(Custom_data_set &scr, ConvNet model);
double classification_accuracy(Custom_data_set &scr, ConvNet model, bool save_error = false);
void train(Custom_data_set &train_data_set, Custom_data_set &val_data_set, ConvNet &model, int epochs, torch::data::DataLoaderOptions OptionsData, torch::Device device = torch::kCPU);
