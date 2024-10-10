# MNIST-CPP-and-python
Beginner-friendly repo on how to Code a Simple Neural network with backprop in C++, bind it to python and train MNIST!  

## Summary  
ReLu and Linear layers are implemented in C++ following  PyTorch's naming convention. Some functionals like softmax are also implemented. The code is bound to python using pybind11. The model is trained on MNIST dataset using a python script.

## How to run  
1. Download the data via `make download_mnist`.  
2. Install the python dependencies via `pip install -r requirements.txt`.
3. Compile the C++ code via `make`.  
4. Run the python script via `python train.py`.
5. (Optional) Compare against pytorch training via `python train_pytorch.py`.