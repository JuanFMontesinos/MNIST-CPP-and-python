// src/dataloader.h
#ifndef DATALOADER_H
#define DATALOADER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

class DataLoader {
public:
    static std::vector<std::vector<float>> load_images(const std::string& filepath);
    static std::vector<int> load_labels(const std::string& filepath);
};

#endif
