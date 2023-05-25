import pathlib
import absl
import jax
import jax.numpy as jnp
import optax

from EasyLM.jax_utils import (
    cross_entropy_loss_and_accuracy,
    set_random_seed,
)
from EasyLM.models.llama.llama_model import (
    LLaMAConfig,
    FlaxLLaMAForCausalLMModule,
)
from EasyLM.checkpoint import StreamingCheckpointer

from iree.jax import Program, store_global

FLAGS = absl.flags.FLAGS

absl.flags.DEFINE_string("dtype", "fp32", "The tensor element type")
absl.flags.DEFINE_string("checkpoint", "fp32", "The tensor element type")
absl.flags.DEFINE_integer("seq_length", 2048, "Sequence length")
absl.flags.DEFINE_string(
    "load_checkpoint",
    "params::"
    + str(pathlib.Path(__file__).parent.parent.resolve())
    + "open_llama_7b_700bt_preview_easylm/"
    + "open_llama_7b_700bt_easylm",
    "Model checkpoint downloaded by running "
    + "git clone git@hf.co:openlm-research/open_llama_7b_700bt_preview_easylm",
)
absl.flags.DEFINE_string(
    "mlir_bytecode_path", "./llama.mlir", "The output MLIR file"
)
absl.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
absl.flags.DEFINE_integer("seed", 42, "RNG seed")


def _create_program():
    # NOTE: Must call set_random_seed to initiialize a global variable
    # required by next_rng.
    set_random_seed(0)

    # Load the trainstate checkpoint which contains only parameters,
    # so the first return value, train_state, should be ignored.
    _, params = StreamingCheckpointer.load_trainstate_checkpoint(
        FLAGS.load_checkpoint, disallow_trainstate=True
    )

    # The checkpoint is bf16, which is not supported by IREE
    # (https://github.com/iree-org/iree-jax/issues/72).
    params = jax.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float16), params)

    # Create the optimizer state given model parameters.
    tx = optax.adam(learning_rate=FLAGS.learning_rate)
    opt_state = tx.init(params)

    # Create model.apply, the function
    #
    # NOTE: LLaMAConfig.get_default_config() returns
    # ml_collections.ConfigDict other than LLaMAConfig.
    llama_config = LLaMAConfig(**LLaMAConfig.get_default_config())
    model = FlaxLLaMAForCausalLMModule(
        llama_config, dtype=jnp.float32, param_dtype=jnp.float16
    )

    def loss(params, input_tokens, target_tokens):
        logits = model.apply(params, input_tokens).logits
        loss, _ = cross_entropy_loss_and_accuracy(logits, target_tokens)
        return loss

    def train_step(params, opt_state, input_tokens, target_tokens):
        grads = jax.grad(loss)(params, input_tokens, target_tokens)
        diff, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, diff)
        return new_params, new_opt_state

    SEQ_SHAPE = (1, FLAGS.seq_length)
    seq_type = jax.core.ShapedArray(SEQ_SHAPE, dtype=jnp.int32)

    class LLaMAProgram(Program):
        _params = Program.export_global(params, initialize=True, mutable=True)
        _opt_state = Program.export_global(
            opt_state, initialize=True, mutable=True
        )

        @Program.kernel
        def _train_step(params, opt_state, input_tokens, target_tokens):
            return train_step(params, opt_state, input_tokens, target_tokens)

        def train_step(self, x=seq_type, y=seq_type):
            new_params, new_opt_state = self._train_step(
                self._params, self._opt_state, x, y
            )
            store_global(self._params, new_params)
            store_global(self._opt_state, new_opt_state)

        def get_params(self):
            return self._params

    return LLaMAProgram


def main(argv):
    with open(FLAGS.mlir_bytecode_path, "wb") as f:
        Program.get_mlir_module(_create_program()).operation.write_bytecode(f)


if __name__ == "__main__":
    absl.app.run(main)
