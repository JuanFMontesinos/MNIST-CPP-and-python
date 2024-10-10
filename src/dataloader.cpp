// src/dataloader.cpp
#include "dataloader.h"

std::vector<std::vector<float>> DataLoader::load_images(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    int magic_number = 0;
    int num_images = 0;
    int rows = 0;
    int cols = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_images, sizeof(num_images));
    file.read((char*)&rows, sizeof(rows));
    file.read((char*)&cols, sizeof(cols));

    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    std::vector<std::vector<float>> images(num_images, std::vector<float>(rows * cols));
    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < rows * cols; ++j) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, sizeof(pixel));
            images[i][j] = pixel / 255.0f;
        }
    }

    return images;
}

std::vector<int> DataLoader::load_labels(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    int magic_number = 0;
    int num_labels = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_labels, sizeof(num_labels));

    magic_number = __builtin_bswap32(magic_number);
    num_labels = __builtin_bswap32(num_labels);

    std::vector<int> labels(num_labels);
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));
        labels[i] = static_cast<int>(label);
    }

    return labels;
}
