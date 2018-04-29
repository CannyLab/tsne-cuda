#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include "common.h"
namespace Data {
	float * load_mnist(std::string fname, int& num_images, int& num_rows, int& num_columns);
	float * load_cifar10(std::string fpath);
	float * load_cifar100(std::string fname);
}



#endif 
