#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include "common.h"
namespace Data {
	thrust::host_vector<float> load_file(std::string fname);
}



#endif 
