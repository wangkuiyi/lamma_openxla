import absl
import jax
import jax.numpy as jnp

from flax.training.train_state import TrainState

from EasyLM.jax_utils import (
    JaxRNG,
    next_rng,
    match_partition_rules,
    cross_entropy_loss_and_accuracy,
    global_norm,
    get_float_dtype_by_name,
    set_random_seed,
    average_metrics,
    get_weight_decay_mask,
    make_shard_and_gather_fns,
)
from EasyLM.models.llama.llama_model import (
    LLaMAConfig,
    FlaxLLaMAForCausalLM,
    FlaxLLaMAForCausalLMModule,
)
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.optimizers import OptimizerFactory

FLAGS = absl.flags.FLAGS

absl.flags.DEFINE_string("dtype", "fp32", "The tensor element type")
absl.flags.DEFINE_string("checkpoint", "fp32", "The tensor element type")
absl.flags.DEFINE_integer("seq_length", 2048, "Sequence length")
absl.flags.DEFINE_string(
    "load_checkpoint",
    "params::/Users/y/w/open_llama_7b_700bt_preview_easylm/open_llama_7b_700bt_easylm",
    "Download checkpoint git clone git@hf.co:openlm-research/open_llama_7b_700bt_preview_easylm",
)


def main(argv):
    # NOTE: LLaMAConfig.get_default_config() returns
    # ml_collections.ConfigDict other than LLaMAConfig.
    llama_config = LLaMAConfig(**LLaMAConfig.get_default_config())

    model = FlaxLLaMAForCausalLMModule(
        llama_config, dtype=get_float_dtype_by_name(FLAGS.dtype)
    )

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        OptimizerFactory.get_default_config(),
        get_weight_decay_mask(LLaMAConfig.get_weight_decay_exclusions()),
    )

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, FLAGS.seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, FLAGS.seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, FLAGS.seq_length), dtype=jnp.int32),
            rngs=rng_generator(llama_config.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    # NOTE: Must call set_random_seed to initiialize a global variable
    # required by next_rng.
    set_random_seed(0)
    train_state_shapes = jax.eval_shape(init_fn, next_rng())

    train_state_partition = match_partition_rules(
        LLaMAConfig.get_partition_rules(), train_state_shapes
    )
    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    checkpointer = StreamingCheckpointer(
        StreamingCheckpointer.get_default_config(),
        "/tmp/",
        enable=True,
    )

    mesh_dim = "1,-1,1"
    with LLaMAConfig.get_jax_mesh(mesh_dim):
        train_state, restored_params = checkpointer.load_trainstate_checkpoint(
            FLAGS.load_checkpoint, train_state_shapes, shard_fns
        )


if __name__ == "__main__":
    absl.app.run(main)
