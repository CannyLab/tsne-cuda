
#include "util/data_utils.h"

float * Data::load_mnist(std::string fname, int& num_images, int& num_rows, int& num_columns) {
	// Construct the file stream
    std::ifstream mnist_data_file(fname, std::ios::in | std::ios::binary);
    
    // Read the data header
    std::cout << "Reading file header..." << std::endl;
    int magic_number = 0;
    mnist_data_file.read(reinterpret_cast<char *>(&magic_number), sizeof(int));
    if (mnist_data_file.gcount() != 4) {std::cout << "File read error (magic number). The number was: " << magic_number << std::endl;}
    magic_number = ((magic_number>>24)&0xff) | // move byte 3 to byte 0
                    ((magic_number<<8)&0xff0000) | // move byte 1 to byte 2
                    ((magic_number>>8)&0xff00) | // move byte 2 to byte 1
                    ((magic_number<<24)&0xff000000); // byte 0 to byte 3
    if (magic_number != 2051) {std::cout << "Invalid magic number. The number was: " << magic_number << std::endl;}
    
    mnist_data_file.read(reinterpret_cast<char *>(&num_images), sizeof(int));
    if (mnist_data_file.gcount() != 4) {std::cout << "File read error (number of images)." << std::endl;}

    num_images = ((num_images>>24)&0xff) | // move byte 3 to byte 0
                    ((num_images<<8)&0xff0000) | // move byte 1 to byte 2
                    ((num_images>>8)&0xff00) | // move byte 2 to byte 1
                    ((num_images<<24)&0xff000000); // byte 0 to byte 3
    std::cout << "Num Images: " << num_images << std::endl;

    mnist_data_file.read(reinterpret_cast<char *>(&num_rows), sizeof(int));
    if (mnist_data_file.gcount() != 4) {std::cout << "File read error (number of rows)." << std::endl;}

    num_rows = ((num_rows>>24)&0xff) | // move byte 3 to byte 0
                    ((num_rows<<8)&0xff0000) | // move byte 1 to byte 2
                    ((num_rows>>8)&0xff00) | // move byte 2 to byte 1
                    ((num_rows<<24)&0xff000000); // byte 0 to byte 3
    std::cout << "Num Rows: " << num_rows << std::endl;

    mnist_data_file.read(reinterpret_cast<char *>(&num_columns), sizeof(int));
    if (mnist_data_file.gcount() != 4) {std::cout << "File read error (number of columns)." << std::endl;}

    num_columns = ((num_columns>>24)&0xff) | // move byte 3 to byte 0
                    ((num_columns<<8)&0xff0000) | // move byte 1 to byte 2
                    ((num_columns>>8)&0xff00) | // move byte 2 to byte 1
                    ((num_columns<<24)&0xff000000); // byte 0 to byte 3
    std::cout << "Num Cols: " << num_columns << std::endl;

    std::cout << "Reading pixels from file..." << std::endl;
    uint8_t pixel_val = 0;
    float* data = new float[num_images*num_rows*num_columns];
    for (int idx = 0; idx < num_images; idx++) {
        for (int jdx = 0; jdx < num_rows*num_columns; jdx++) {
            mnist_data_file.read(reinterpret_cast<char *>(&pixel_val), sizeof(uint8_t));
            if (mnist_data_file.gcount() != 1) {std::cout << "File read error (pixel)." << std::endl;}
            data[idx*num_rows*num_columns + jdx] = ((float) pixel_val) / 255.0f;
        }
    }

    std::cout << "Done reading!" << std::endl;
    mnist_data_file.close();
    return data;
}

float * Data::load_cifar10(std::string fpath) {
	int num_images = 50000;
	    
	int num_channels = 3;
	int num_rows = 32;
	int num_columns = 32;

	int bytes_per_image = num_channels * num_rows * num_columns;

	float* data = new float[num_images*bytes_per_image];

	for (int i = 0; i < 5; i++) {
		char bin_file_name[50];
		snprintf(bin_file_name, sizeof(bin_file_name), "/data_batch_%d.bin", i + 1);

		std::string fname = fpath;
		fname.append(bin_file_name);

		// Construct the file stream
	    std::ifstream cifar_data_file(fname, std::ios::in | std::ios::binary);
	   
	    
	    int num_images_batch = 10000;
	    int batch_offset = num_images_batch * i;
	    
	  
	    std::cout << "Reading pixels from file " << i + 1 << "..." << std::endl;
	    uint8_t pixel_val = 0;
	    
	    for (int idx = 0; idx < num_images_batch; idx++) {
	    	int file_idx = idx + batch_offset;
	  		cifar_data_file.ignore(1); // Ignore the label byte at the beginning of each file
	        for (int jdx = 0; jdx < bytes_per_image; jdx++) {
	            cifar_data_file.read(reinterpret_cast<char *>(&pixel_val), sizeof(uint8_t));
	            data[file_idx*bytes_per_image + jdx] = ((float) pixel_val) / 255.0f;
	        }
	    }

	    std::cout << "Done reading!" << std::endl;
	    cifar_data_file.close();
	}
	return data;
}

float * Data::load_cifar100(std::string fname) {
	int num_images = 50000;
	    
	int num_channels = 3;
	int num_rows = 32;
	int num_columns = 32;

	int bytes_per_image = num_channels * num_rows * num_columns;

	float* data = new float[num_images*bytes_per_image];


	// Construct the file stream
    std::ifstream cifar_data_file(fname, std::ios::in | std::ios::binary);
  
    std::cout << "Reading pixels from file..." << std::endl;
    uint8_t pixel_val = 0;
    
    for (int idx = 0; idx < num_images; idx++) {
  		cifar_data_file.ignore(2); // Ignore the label byte at the beginning of each file
        for (int jdx = 0; jdx < bytes_per_image; jdx++) {
            cifar_data_file.read(reinterpret_cast<char *>(&pixel_val), sizeof(uint8_t));
            data[idx*bytes_per_image + jdx] = ((float) pixel_val) / 255.0f;
        }
    }

    std::cout << "Done reading!" << std::endl;
    cifar_data_file.close();
	
	return data;
}

void Data::save(std::string fname, const float * const points, const unsigned int N, const unsigned int NDIMS) {
    std::ofstream save_file(fname, std::ios::out | std::ios::binary);
    save_file << N << NDIMS;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < NDIMS; j++) {
            save_file << points[i * NDIMS + j];
        }
    }
    save_file.close();
}

void Data::save(std::string fname, thrust::device_vector<float> points, const unsigned int N, const unsigned int NDIMS) {
    float *data = new float[N * NDIMS];
    thrust::copy(points.begin(), points.end(), data);
    Data::save(fname, data, N, NDIMS);
}

float * Data::load(std::string fname) {
    std::ifstream load_file(fname, std::ios::in | std::ios::binary);
    unsigned int N;
    unsigned int NDIMS;
    float pt;

    load_file.read(reinterpret_cast<char *>(&N), sizeof(unsigned int));
    load_file.read(reinterpret_cast<char *>(&NDIMS), sizeof(unsigned int));
    float *data = new float[N * NDIMS];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < NDIMS; j++) {
            load_file.read(reinterpret_cast<char *>(&pt), sizeof(unsigned int));
            data[i * NDIMS + j] = pt;
        }
    }

    load_file.close();
    return data;
}
