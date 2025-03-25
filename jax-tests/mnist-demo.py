import flax.nnx as nnx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tensorflow.keras.datasets as dts

# Define neural network
class SimpleNN(nnx.Module):
  def __init__(self, n_features: int = 64, n_hidden: int = 100, n_targets: int = 10,
               *, rngs: nnx.Rngs):
    self.n_features = n_features
    self.layer1 = nnx.Linear(n_features, n_hidden, rngs=rngs)
    self.layer2 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
    self.layer3 = nnx.Linear(n_hidden, n_targets, rngs=rngs)

  def __call__(self, x):
    x = x.reshape(x.shape[0], self.n_features) # Flatten images.
    x = nnx.selu(self.layer1(x))
    x = nnx.selu(self.layer2(x))
    x = self.layer3(x)
    return x

# Fetch MNIST dataset
(x_train_np, y_train_np), (x_test_np, y_test_np) = dts.mnist.load_data()

# Convert to JAX arrays
x_train, y_train, x_test, y_test = map(jnp.asarray, [x_train_np, y_train_np, x_test_np, y_test_np])

# Create model
model = SimpleNN(rngs=nnx.Rngs(0))
