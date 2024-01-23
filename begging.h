#pragma once
#include "data_set.h"
#include "model.h"


void begging(Custom_data_set &train_data_set, Custom_data_set &val_data_set, int amt_model, int epochs, torch::data::DataLoaderOptions OptionsData, torch::Device device = torch::kCPU);
void save_begging_model(std::vector<ConvNet> &vector_model, std::string path);
std::vector<ConvNet> load_begging_model(std::string path);
double accuracy_begging_model(Custom_data_set &scr, std::vector<ConvNet> &vector_model);