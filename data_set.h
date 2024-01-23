#pragma once

#include <vector>
#include <sstream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>


struct Element
{
	Element() {};
	Element(std::string img, int label) :img{ img }, label{ label } {};

	std::string img;
	int label;
};


class Custom_data_set : public torch::data::Dataset<Custom_data_set>
{
    private:
		std::vector<Element> _csv;

    public:
		Custom_data_set(std::string& file_names_csv);
		Custom_data_set(std::vector<Element> data);

		torch::data::Example<> get(size_t index) override;
		Element get_element(size_t index);
		torch::optional<size_t> size() const override;

		std::vector<Custom_data_set> make_bootstrap_data_set(size_t amt_data_set, bool intersection = false);
		void save_data_set(std::string file_names_csv);
};


std::vector<Element> read_csv(std::string& location);
torch::Tensor img_to_tensor(cv::Mat scr);
torch::Tensor img_to_tensor(std::string path);