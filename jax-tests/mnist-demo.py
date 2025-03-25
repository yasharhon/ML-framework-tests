import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
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

# Define loss function
def loss_fun(
    model: nnx.Module,
    data: jax.Array,
    labels: jax.Array
):
    logits = model(data)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    ).mean()
    return loss, logits

# Define training step with JIT compilation
@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    data: jax.Array,
    labels: jax.Array
):
    loss_gradient = nnx.grad(loss_fun, has_aux=True)  # gradient transform!
    grads, logits = loss_gradient(model, data, labels)
    optimizer.update(grads)  # inplace update

# Fetch MNIST dataset
(x_train_np, y_train_np), (x_test_np, y_test_np) = dts.mnist.load_data()

# Convert to JAX arrays
x_train, y_train, x_test, y_test = map(jnp.asarray, [x_train_np, y_train_np, x_test_np, y_test_np])

# Create model
model = SimpleNN(rngs=nnx.Rngs(0))

# Create optimizer
optimizer = nnx.Optimizer(model, optax.sgd(learning_rate=0.05))
