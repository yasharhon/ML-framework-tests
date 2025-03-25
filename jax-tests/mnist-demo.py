import tensorflow.keras.datasets as dts
import matplotlib.pyplot as plt

# Fetch MNIST dataset
(x_train, y_train), (x_test, y_test) = dts.mnist.load_data()
