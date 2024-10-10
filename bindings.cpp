// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "include/modules.h"
#include "include/activations.h"
#include "include/functionals.h"
#include "include/cross_entropy.h"
#include "include/dataloader.h"

namespace py = pybind11;

PYBIND11_MODULE(libmnist, m) {
    // Bind the ReLu class
    py::class_<ReLu>(m, "ReLu")
        .def(py::init<>())
        .def("forward", [](ReLu& self, const std::vector<float>& input) {
            auto output = self.forward(input);
            // Return as NumPy array
            return py::array_t<float>(output.size(), output.data());
        }, "Apply ReLu activation")
        .def("backward", [](ReLu& self, const std::vector<float>& grad_output) {
            auto grad_input = self.backward(grad_output);
            return py::array_t<float>(grad_input.size(), grad_input.data());
        }, "Compute the backward pass of ReLu")
        .def("update", &ReLu::update, py::arg("lr"), "Update the parameters of ReLu");

    // Bind the LinearLayer class
    py::class_<LinearLayer>(m, "LinearLayer")
        .def(py::init<int, int>())
        .def("forward", [](LinearLayer& self, const std::vector<float>& input) {
            auto output = self.forward(input);
            // Return as NumPy array
            return py::array_t<float>(output.size(), output.data());
        }, "Perform forward pass with Linear Layer")
        .def("backward", [](LinearLayer& self, const std::vector<float>& grad_output) {
            auto grad_input = self.backward(grad_output);
            return py::array_t<float>(grad_input.size(), grad_input.data());
        }, "Compute the backward pass of Linear Layer")
        .def("update", &LinearLayer::update, py::arg("lr"), "Update the parameters of LinearLayer")
        .def_readwrite("weights", &LinearLayer::weights)
        .def_readwrite("bias", &LinearLayer::bias)
        .def_readwrite("grad_weights", &LinearLayer::grad_weights)
        .def_readwrite("grad_bias", &LinearLayer::grad_bias);  


    // Bind the SoftmaxndCrossEntropy class
    py::class_<SoftmaxndCrossEntropy>(m, "SoftmaxndCrossEntropy")
        .def(py::init<int>())
        .def("forward", [](SoftmaxndCrossEntropy& self, const std::vector<float>& input, int class_label) {
            return self.forward(input, class_label);
        }, "Compute the forward pass of Softmax and Cross Entropy")
        .def("backward", [](SoftmaxndCrossEntropy& self) {
            auto grad = self.backward();
            return py::array_t<float>(grad.size(), grad.data());
        }, "Compute the backward pass of Softmax and Cross Entropy");
    // Bind the DataLoader class
    py::class_<DataLoader>(m, "DataLoader")
        .def_static("load_images", [](const std::string& filepath) {
            auto images = DataLoader::load_images(filepath);
            py::ssize_t num_images = static_cast<py::ssize_t>(images.size());
            if (num_images == 0) {
                throw std::runtime_error("No images loaded");
            }
            py::ssize_t image_size = static_cast<py::ssize_t>(images[0].size());

            // Create a NumPy array of shape (num_images, image_size)
            py::array_t<float> result({num_images, image_size});

            auto buf = result.mutable_unchecked<2>();

            for (py::ssize_t i = 0; i < num_images; ++i) {
                if (static_cast<py::ssize_t>(images[i].size()) != image_size) {
                    throw std::runtime_error("Inconsistent image sizes");
                }
                for (py::ssize_t j = 0; j < image_size; ++j) {
                    buf(i, j) = images[i][j];
                }
            }
            return result;
        }, "Load images from file")
        .def_static("load_labels", [](const std::string& filepath) {
            auto labels = DataLoader::load_labels(filepath);
            py::ssize_t num_labels = static_cast<py::ssize_t>(labels.size());

            py::array_t<int> result({num_labels});
            auto buf = result.mutable_unchecked<1>();
            for (py::ssize_t i = 0; i < num_labels; ++i) {
                buf(i) = labels[i];
            }
            return result;
        }, "Load labels from file");

    // Bind the functionals submodule
    py::module_ functionals = m.def_submodule("functionals", "Submodule for functional operations");
    functionals.def("softmax", [](const std::vector<float>& input) {
        auto output = functionals::softmax(input);
        return py::array_t<float>(output.size(), output.data());
    }, "Compute the softmax of a 1D vector");
    functionals.def("flatten2d", [](const std::vector<std::vector<float>>& input) {
        auto output = functionals::flatten2d(input);
        return py::array_t<float>(output.size(), output.data());
    }, "Flatten a 2D vector into a 1D vector");
}

