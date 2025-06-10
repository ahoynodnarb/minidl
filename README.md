minidl is a miniature deep learning framework built mostly in pure Python and NumPy/CuPy. This project was meant for me to get a better idea at how larger, more abstracted frameworks like PyTorch and Tensorflow work under the hood.

This originally started as my Linear Algebra Honors Project, which focused on the importance of linear algebra in performant and scalable deep learning. Since then, the core architecture has stayed the same, but I've tried to add a bit of polish here and there.

The CNN's detailed in `examples/MNIST_Classifier.py` and `examples/cifar10.py` were able to reach 98.5% and ~81% accuracy on their respective testing datasets after training over 50 epochs for 10-15 minutes each.