/**
 * @brief Implementation file of the data_utils.h header
 * 
 * @file data_utils.cu
 * @author your name
 * @date 2018-05-05
 * Copyright (c) 2018, Regents of the University of Californias
 */

#include "include/util/data_utils.h"

float* tsnecuda::util::LoadMnist(std::string file_name, int32_t& num_images,
        int32_t& num_rows, int32_t& num_columns) {
    // Construct the file stream
    std::ifstream mnist_file(file_name, std::ios::in | std::ios::binary);

    // Read the data header
    int32_t magic_number = 0;
    mnist_file.read(reinterpret_cast<char*>(&magic_number), sizeof(int32_t));
    if (mnist_file.gcount() != 4) {
        std::cout << "E: Magic number: " << magic_number << std::endl;
    }

    magic_number = ((magic_number >> 24)&0xff) |  // move byte 3 to byte 0
            ((magic_number << 8)&0xff0000) |  // move byte 1 to byte 2
            ((magic_number >> 8)&0xff00) |  // move byte 2 to byte 1
            ((magic_number << 24)&0xff000000);  // byte 0 to byte 3
    if (magic_number != 2051) {
        std::cout << "E: Magic number: " << magic_number << std::endl;
    }

    mnist_file.read(reinterpret_cast<char*>(&num_images), sizeof(int32_t));
    if (mnist_file.gcount() != 4) {
        std::cout << "E: Number of images." << std::endl;
    }

    num_images = ((num_images >> 24)&0xff) |  // move byte 3 to byte 0
            ((num_images << 8)&0xff0000) |  // move byte 1 to byte 2
            ((num_images >> 8)&0xff00) |  // move byte 2 to byte 1
            ((num_images << 24)&0xff000000);  // byte 0 to byte 3
    std::cout << "Num Images: " << num_images << std::endl;

    mnist_file.read(reinterpret_cast<char *>(&num_rows), sizeof(int32_t));
    if (mnist_file.gcount() != 4) {
        std::cout << "E: Number of rows." << std::endl;
    }

    num_rows = ((num_rows >> 24)&0xff) |  // move byte 3 to byte 0
            ((num_rows << 8)&0xff0000) |  // move byte 1 to byte 2
            ((num_rows >> 8)&0xff00) |  // move byte 2 to byte 1
            ((num_rows << 24)&0xff000000);  // byte 0 to byte 3

    mnist_file.read(reinterpret_cast<char *>(&num_columns), sizeof(int32_t));
    if (mnist_file.gcount() != 4) {
        std::cout << "E: Number of Columns." << std::endl;
    }

    num_columns = ((num_columns >> 24)&0xff) |  // move byte 3 to byte 0
            ((num_columns << 8)&0xff0000) |  // move byte 1 to byte 2
            ((num_columns >> 8)&0xff00) |  // move byte 2 to byte 1
            ((num_columns << 24)&0xff000000);  // byte 0 to byte 3

    uint8_t pixel_val = 0;
    float* data = new float[num_images*num_rows*num_columns];
    for (int idx = 0; idx < num_images; idx++) {
        for (int jdx = 0; jdx < num_rows*num_columns; jdx++) {
            mnist_file.read(reinterpret_cast<char*>(&pixel_val),
                            sizeof(uint8_t));
            if (mnist_file.gcount() != 1) {
                std::cout << "E: File read error (pixel)." << std::endl;
            }
            data[idx*num_rows*num_columns + jdx] = static_cast<float>(pixel_val) / 255.0f;
        }
    }

    std::cout << "Done reading!" << std::endl;
    mnist_file.close();
    return data;
}

float* tsnecuda::util::LoadCifar10(std::string file_path) {
    int32_t kNumImages = 50000;

    int32_t kNumChannels = 3;
    int32_t kNumRows = 32;
    int32_t kNumColumns = 32;

    int32_t bytes_per_image = kNumChannels * kNumRows * kNumColumns;

    float* data = new float[kNumImages*bytes_per_image];

    for (size_t i = 0; i < 5; i++) {
        char binary_file_name[50];
        snprintf(binary_file_name, sizeof(binary_file_name),
            "/data_batch_%d.bin", i + 1);

        std::string file_name = file_path;
        file_name.append(binary_file_name);

        // Construct the file stream
        std::ifstream cifar_data_file(file_name,
            std::ios::in | std::ios::binary);


        size_t kNumImagesBatch = 10000;
        size_t batch_offset = kNumImagesBatch * i;


        std::cout << "Reading pixels from file " << i + 1 << "..." << std::endl;
        uint8_t pixel_val = 0;

        for (size_t idx = 0; idx < kNumImagesBatch; idx++) {
            int32_t file_idx = idx + batch_offset;
            cifar_data_file.ignore(1);  // Ignore the label byte
                                        // at the beginning of each file
            for (size_t jdx = 0; jdx < bytes_per_image; jdx++) {
                cifar_data_file.read(reinterpret_cast<char*>(&pixel_val),
                                    sizeof(uint8_t));
                data[file_idx*bytes_per_image + jdx] = static_cast<float>(pixel_val) / 255.0f;
            }
        }

        std::cout << "Done reading!" << std::endl;
        cifar_data_file.close();
    }
    return data;
}

float* tsnecuda::util::LoadCifar100(std::string file_name) {
    int32_t kNumImages = 50000;

    int32_t kNumChannels = 3;
    int32_t kNumRows = 32;
    int32_t kNumColumns = 32;

    int32_t bytes_per_image = kNumChannels * kNumRows * kNumColumns;

    float* data = new float[kNumImages*bytes_per_image];


    // Construct the file stream
    std::ifstream cifar_data_file(file_name, std::ios::in | std::ios::binary);

    uint8_t pixel_val = 0;
    for (size_t idx = 0; idx < kNumImages; idx++) {
            // Ignore the label byte at the beginning of each file
          cifar_data_file.ignore(2);
        for (size_t jdx = 0; jdx < bytes_per_image; jdx++) {
            cifar_data_file.read(reinterpret_cast<char*>(&pixel_val),
                sizeof(uint8_t));
            data[idx*bytes_per_image + jdx] = static_cast<float>(pixel_val) / 255.0f;
        }
    }

    std::cout << "Done reading!" << std::endl;
    cifar_data_file.close();

    return data;
}

void tsnecuda::util::Save(const float * const points,
        std::string file_name,  const int num_points,
        const int num_dims) {
    std::ofstream save_file(file_name, std::ios::out | std::ios::binary);
    save_file << num_points << num_dims;
    for (size_t i = 0; i < num_points; i++) {
        for (size_t j = 0; j < num_dims; j++) {
            save_file << points[i * num_dims + j];
        }
    }
    save_file.close();
}

void tsnecuda::util::Save(thrust::device_vector<float> d_points,
        std::string file_name,  const int num_points,
        const int num_dims) {
    float *data = new float[num_points * num_dims];
    thrust::copy(d_points.begin(), d_points.end(), data);
    tsnecuda::util::Save(data, file_name, num_points, num_dims);
    delete[] data;
}

float* tsnecuda::util::Load(std::string file_name) {
    std::ifstream load_file(file_name, std::ios::in | std::ios::binary);
    int num_points;
    int num_dims;
    float kPoint = 0.0f;

    load_file.read(reinterpret_cast<char *>(&num_points), sizeof(int));
    load_file.read(reinterpret_cast<char *>(&num_dims), sizeof(int));
    float *data = new float[num_points * num_dims];
    for (size_t i = 0; i < num_points; i++) {
        for (size_t j = 0; j < num_dims; j++) {
            load_file.read(reinterpret_cast<char*>(&kPoint), sizeof(int));
            data[i * num_dims + j] = kPoint;
        }
    }
    load_file.close();
    return data;
}
