"""Define a linear regression model."""
import absl  # type: ignore
import jax
import jax.numpy as jnp
import optax  # type: ignore
import flax
from typing import Tuple, Callable, Dict, Any, Union
from tqdm import tqdm  # type: ignore

# The return type of flax.linen.Module.init()
# https://tinyurl.com/2p87bvt6
from flax.core.scope import FrozenVariableDict


FLAGS = absl.flags.FLAGS

absl.flags.DEFINE_float("learning_rate", 0.01, "learning rate")

INPUT_SHAPE = (1, 4)
OUTPUT_SHAPE = (1, 1)


class _SumUpInputVector(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, x):
        x = flax.linen.Dense(features=1, use_bias=False)(x)
        return x


ModelState = Union[FrozenVariableDict, Dict[str, Any]]


def model(rng) -> Tuple[ModelState, Callable]:
    model = _SumUpInputVector()
    example_input = jnp.ones(INPUT_SHAPE)
    model_params = model.init(rng, example_input)
    return (model_params, model.apply)


def optimizer(model_params) -> Tuple[Tuple, Callable]:
    opt = optax.adam(learning_rate=FLAGS.learning_rate)
    opt_states = opt.init(model_params)
    return (opt_states, opt.update)


def init_training(rng):
    rng, subkey = jax.random.split(rng)
    model_params, forward = model(subkey)
    opt_states, update = optimizer(model_params)

    def loss(model_params, x, y):
        return optax.squared_error(forward(model_params, x), y).mean()

    @jax.jit
    def step(model_params, opt_states, x, y):
        grads = jax.grad(loss)(model_params, x, y)
        diff, opt_states = update(grads, opt_states, model_params)
        model_params = optax.apply_updates(model_params, diff)
        return model_params, opt_states

    return model_params, opt_states, forward, step, rng


class Tests(absl.testing.absltest.TestCase):
    def test_train_with_synthetic_data(self):
        rng = jax.random.PRNGKey(0)
        model_params, opt_states, forward, step, rng = init_training(rng)

        for _ in tqdm(range(2000)):
            rng, subkey = jax.random.split(rng)
            x = jax.random.uniform(subkey, shape=INPUT_SHAPE)
            y = jnp.array([[x.sum()]])
            model_params, opt_states = step(model_params, opt_states, x, y)

        # The learned parameters are expected to be [1,1,1,1], which, when
        # mutiplied with x, gets x.sum().
        weights = model_params["params"]["Dense_0"]["kernel"]
        self.assertTrue(
            jnp.allclose(weights, jnp.asarray([1.0, 1.0, 1.0, 1.0]), rtol=1e-2)
        )

        p = forward(model_params, jnp.asarray([0, 1, 2, 3]))
        self.assertTrue(jnp.allclose(p, jnp.asarray([6.0]), rtol=1e-2))


if __name__ == "__main__":
    absl.testing.absltest.main()
