
#include "util/data_utils.h"

thrust::host_vector<float> Data::load_file(std::string fname) {
	thrust::host_vector<float> data_vec(100);
	std::ifstream data_file;
	data_file.open(fname);
	float buf;
	while (data_file >> buf)
		data_vec.push_back(buf);
	return data_vec;
}