"""Export the GPT-2 model and include all required coefficients."""

import linear_regression
import absl.app
import absl.flags
import jax
import jax.numpy as jnp
from iree.jax import Program, store_global

FLAGS = absl.flags.FLAGS


def _create_mlir():
    input_type = jax.core.ShapedArray(linear_regression.INPUT_SHAPE, dtype=jnp.float32)
    output_type = jax.core.ShapedArray(
        linear_regression.OUTPUT_SHAPE, dtype=jnp.float32
    )

    rng = jax.random.PRNGKey(0)
    model_params, opt_states, forward, step, rng = linear_regression.init_training(rng)

    # The generated MLIR module name will be the prefix before Program.
    class LinearRegressionProgram(Program):
        _model_params = Program.export_global(
            model_params, initialize=True, mutable=True
        )
        _opt_states = Program.export_global(opt_states, initialize=True, mutable=True)

        @Program.kernel
        def _predict(model_params, x):
            return forward(model_params, x)

        def predict(self, x=input_type):
            return self._predict(self._model_params, x)

        @Program.kernel
        def _train_step(model_params, opt_states, x, y):
            return step(model_params, opt_states, x, y)

        def train_step(self, x=input_type, y=output_type):
            new_model_params, new_opt_states = self._train_step(
                self._model_params, self._opt_states, x, y
            )
            store_global(self._model_params, new_model_params)
            store_global(self._opt_states, new_opt_states)

        def get_params(self):
            return self._model_params

    return LinearRegressionProgram


absl.flags.DEFINE_string("ir_path", "./lr.mlir", "The output MLIR file")


def main(argv):
    mlir_module = _create_mlir()

    with open(FLAGS.ir_path, "w") as f:
        f.write(str(Program.get_mlir_module(mlir_module)))


if __name__ == "__main__":
    absl.app.run(main)
