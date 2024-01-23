#include "begging.h"
#include <filesystem>
#include "utils.h"


void begging(Custom_data_set &train_data_set, Custom_data_set &val_data_set, int amt_model, int epochs, torch::data::DataLoaderOptions OptionsData, torch::Device device) {
	std::vector<Custom_data_set> begging_train_data_set = train_data_set.make_bootstrap_data_set(amt_model);
	std::vector<Custom_data_set> begging_val_data_set = val_data_set.make_bootstrap_data_set(amt_model);

	std::vector<ConvNet> vector_model;
	vector_model.resize(amt_model);

	for (int i = 0; i < amt_model; i++) {
		std::cout << "Model #" << i + 1 << "/" << amt_model << std::endl;

		train(begging_train_data_set.at(i), begging_val_data_set.at(i), vector_model.at(i), epochs, OptionsData);

		std::string name_dataset = "..\\begging_model\\dataset_" + std::to_string(i + 1) + ".csv";

		begging_val_data_set.at(i).save_data_set(name_dataset);

		std::string name_model = "..\\begging_model\\begging_model_" + std::to_string(i + 1) + ".model";
		torch::save(vector_model.at(i), name_model);
	}
}


void save_begging_model(std::vector<ConvNet> &vector_model, std::string path) {
	for (int i = 0; i < vector_model.size(); i++) {
		std::string name_model = "..\\" + path + "\\begging_model_" + std::to_string(i + 1) + ".model";
		torch::save(vector_model.at(i), name_model);
	}
}


std::vector<ConvNet> load_begging_model(std::string path) {
	std::vector<ConvNet> out;
	std::string extension = ".model";
	std::filesystem::directory_iterator iterator("..\\" + path);
	
	for (; iterator != std::filesystem::end(iterator); iterator++)
	{
		if (iterator->path().extension() == extension)
		{
			ConvNet buf;
			torch::load(buf, iterator->path().string());
			out.push_back(buf);
		}
	}

	std::cout << "Load model: " << out.size() << std::endl;
	return out;
}


double accuracy_begging_model(Custom_data_set &scr, std::vector<ConvNet> &vector_model) {
	int error = 0;
	for (int i = 0; i < scr.size().value(); i++) {
		auto obj = scr.get(i);

		std::vector<int> voting;
		voting.resize(2);
		int result = 0;

		for (ConvNet model:vector_model) {
			voting.at(classification(obj.data, model).item<int>())++;
		}
		result = distance(voting.begin(), max_element(voting.begin(), voting.end()));

		//for (auto vote : voting) 
			//std::cout << vote << " ";
		//std::cout << result << std::endl;

		if (result != obj.target.item<int>())
			error++;
	}

	return (double)error / scr.size().value();
}