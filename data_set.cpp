#include "data_set.h"
#include <fstream>
#include <random>

torch::Tensor img_to_tensor(cv::Mat src) {
	cv::cvtColor(src, src, cv::COLOR_BGR2RGB); // camera out - RGB, openCV - BGR

	cv::Mat img_fo_NN = src(cv::Rect(0, 0, src.size().width, src.size().height / 2)).clone();

	cv::Mat mask;
	cv::cvtColor(img_fo_NN, mask, CV_RGB2GRAY);
	cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);

	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(mask, contours, CV_RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	cv::Rect area_attention = cv::boundingRect(contours[0]);
	img_fo_NN = img_fo_NN(area_attention).clone();

	cv::resize(img_fo_NN, img_fo_NN, { 100,100 });

	//cv::imshow("", img_fo_NN);
	//cv::waitKey();


	torch::Tensor img_tensor = torch::from_blob(img_fo_NN.data, { img_fo_NN.rows, img_fo_NN.cols, 3 }, torch::kByte).clone();

	img_tensor = img_tensor.toType(torch::kFloat);
	img_tensor = img_tensor.div(255);
	img_tensor = img_tensor.permute({ 2,0,1 });
	return img_tensor;
}


torch::Tensor img_to_tensor(std::string path) {
	cv::Mat img = cv::imread(path);
	return img_to_tensor(img);
}


std::vector<Element> read_csv(std::string& location) {
	std::fstream in(location, std::ios::in);
	std::string line;
	std::string name;
	std::string label;
	std::vector<Element> csv;

	while (getline(in, line))
	{
		std::stringstream s(line);
		getline(s, name, ',');
		getline(s, label, ',');

		csv.push_back(Element(name, stoi(label)));
	}

	return csv;
}


Custom_data_set::Custom_data_set(std::string& file_names_csv) {
	_csv = read_csv(file_names_csv);
}


Custom_data_set::Custom_data_set(std::vector<Element> data) {
	_csv = data;
}


torch::data::Example<> Custom_data_set::get(size_t index) {

	std::string file_location = _csv[index].img;
	int64_t label = _csv[index].label;

	torch::Tensor img_tensor = img_to_tensor(file_location);

	torch::Tensor label_tensor = torch::full({ 1 }, label);
	label_tensor.to(torch::kInt64);

	return { img_tensor, label_tensor };
}


Element Custom_data_set::get_element(size_t index) {
	return _csv[index];
}


torch::optional<size_t> Custom_data_set::size() const{
	return _csv.size();
};


std::vector<Custom_data_set> Custom_data_set::make_bootstrap_data_set(size_t amt_data_set, bool intersection) {
	std::vector<Custom_data_set> out;
	std::vector<size_t> new_index_data_set;

	std::vector<std::vector<size_t>> label_index;
	label_index.resize(2);

	for (int i = 0;  i < _csv.size(); i++) {
		if (_csv.at(i).label == 0)
			label_index.at(0).push_back(i);
		else
			label_index.at(1).push_back(i);
	}

	size_t min_size_label = std::min(label_index.at(0).size(), label_index.at(1).size());

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dist_label_0(0, label_index.at(0).size() - 1);
	std::uniform_int_distribution<> dist_label_1(0, label_index.at(1).size() - 1);

	for (int i = 0; i < amt_data_set; i++) {
		if (intersection) {
			for (int j = 0; j < min_size_label; j++) {
				size_t index;

				if (j < min_size_label/2) {
					index = dist_label_0(gen);
					new_index_data_set.push_back(label_index.at(0).at(index));
				}
				else {
					index = dist_label_1(gen);
					new_index_data_set.push_back(label_index.at(1).at(index));
				}
			}
		}
		else {
			int size_label_0 = label_index.at(0).size() / amt_data_set;
			int size_label_1 = label_index.at(1).size() / amt_data_set;

			std::copy(label_index.at(0).begin() + i * size_label_0, label_index.at(0).begin() + i * size_label_0 + size_label_0, std::back_inserter(new_index_data_set));
			std::copy(label_index.at(1).begin() + i * size_label_1, label_index.at(1).begin() + i * size_label_1 + size_label_1, std::back_inserter(new_index_data_set));
		}
	}

	for (int i = 0; i < amt_data_set; i++) {
		std::vector<Element> buf;

		for (int j = 0; j < new_index_data_set.size() / amt_data_set; j++) {
			//std::cout << "i: " << i << "j" << j << std::endl;

			buf.push_back(_csv.at(new_index_data_set.at(i * (new_index_data_set.size() / amt_data_set) + j)));
		}

		out.push_back(Custom_data_set(buf));
	}

	return out;
}

void  Custom_data_set::save_data_set(std::string file_names_csv) {
	std::ofstream out;
	out.open(file_names_csv);
	if (out.is_open())
	{
		for (Element element:_csv) {
			out << element.img << "," << element.label << std::endl;
		}
	}
	out.close();
}

