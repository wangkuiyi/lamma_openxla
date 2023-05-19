"""Export the GPT-2 model and include all required coefficients."""
import inspect
import absl  # type: ignore
import jax
import jax.numpy as jnp
import optax  # type: ignore
import flax
from typing import Tuple, Callable, Dict, Any, Union
from tqdm import tqdm  # type: ignore

FLAGS = absl.flags.FLAGS

absl.flags.DEFINE_float("learning_rate", 0.01, "learning rate")


class MyModel(flax.linen.Module):
    INPUT_SHAPE = [1, 4]

    @flax.linen.compact
    def __call__(self, x):
        x = flax.linen.Dense(features=1, use_bias=False)(x)
        return x


# The return type of flax.linen.Module.init() https://github.com/google/flax/blob/380fef6db994b55f217df9421924a667fc829d56/flax/linen/module.py#LL1559C25-L1559C67
from flax.core.scope import FrozenVariableDict

ModelState = Union[FrozenVariableDict, Dict[str, Any]]


def model(rng) -> Tuple[ModelState, Callable]:
    model = MyModel()
    print(
        f"A flax.linen model is derived from {inspect.getmro(type(model))}, which is no more than a config"
    )

    example_input = jnp.ones(MyModel.INPUT_SHAPE)
    model_state = model.init(rng, example_input)
    print(
        f"The materialized and randomly initialized model state is of type {type(model_state)}"
    )

    return (model_state, model.apply)


def optimizer(model_params) -> Tuple[Tuple, Callable]:
    opt = optax.adam(learning_rate=FLAGS.learning_rate)
    print(f"The function optax.adam returns a named tuple {type(opt)}")

    opt_state = opt.init(model_params)
    print(
        f"Calling {type(opt)}.init to create the optimizer state of type {type(opt_state)}"
    )

    return (opt_state, opt.update)


def main(argv):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    model_state, forward = model(subkey)
    print(model_state)

    opt_state, update = optimizer(model_state)
    print(opt_state)

    def loss(model_state, x, y):
        return optax.squared_error(forward(model_state, x), y).mean()

    @jax.jit
    def step(model_state, opt_state, x, y):
        grads = jax.grad(loss)(model_state, x, y)
        diff, opt_state = update(grads, opt_state, model_state)
        model_state = optax.apply_updates(model_state, diff)
        return model_state, opt_state

    for _ in tqdm(range(2000)):
        key, subkey = jax.random.split(key)
        x = jax.random.uniform(subkey, shape=MyModel.INPUT_SHAPE)
        y = jnp.array([[x.sum()]])
        model_state, opt_state = step(model_state, opt_state, x, y)

    # The learned parameters are expected to be [1,1,1,1], which, when
    # mutiplied with x, gets x.sum().
    print(model_state)


if __name__ == "__main__":
    absl.app.run(main)
