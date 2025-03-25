import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import sklearn.datasets as skldts
import sklearn.model_selection as sklms

# Define neural network
class SimpleNN(nnx.Module):
  def __init__(self, n_features: int = 8*8, n_hidden: int = 100, n_targets: int = 10,
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

# Fetch digits dataset
digits = skldts.load_digits()

# Split into training and test data
splits = sklms.train_test_split(digits.images, digits.target, random_state=0)

# Convert to JAX arrays
x_train, x_test, y_train, y_test = map(jnp.asarray, splits)

# Create model
model = SimpleNN(rngs=nnx.Rngs(0))

# Create optimizer
optimizer = nnx.Optimizer(model, optax.sgd(learning_rate=0.05))

for i in range(101):
    train_step(model, optimizer, x_train, y_train)
    if i % 10 == 1:
      loss, _ = loss_fun(model, x_test, y_test)
      print("Epoch {0}: Loss {1}".format(i, loss))
