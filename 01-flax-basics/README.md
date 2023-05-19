This example demonstrates the essential ideas of Flax and how they work.

Object-oriented thinking is liked by researchers because it makes sense: (1) A model is made up of modules. (2) Each module has a forward method that takes inputs as its parameters and takes model parameters as class data members.

PyTorch's `torch.nn.Module` is a direct translation of the above mental model.  But since we're using a Python function to represent the forward pass, we'll need the Python interpreter to run training and inference programs.  Python interpreter is slow, and there is no Python interpreter on mobile devices.

JAX makes it easy to trace Python code into an intermediate language (IR) and then compile that IR into native code.  As a result, the Python interpreter is no longer required for deep learning program execution. To use the convenience, users must define the forward pass as a **pure function**, which can't access model parameters as class data members. Instead, they must take inputs and model parameters as function parameters.

Researchers usually don't like to write pure functions. The `flax.linen` package has modules that look like those in `torch.nn` but are different in important ways.

1. A `flax.linen` class includes only a module's configuration information. It doesn't have any data members that represent model parameters. Instead, users must use `flax.linen.Module.init(rng, example_input)` to create and randomly initialize the model parameters.

1. A `flax.linen` class has a method called `flax.linen.Module.apply(model_params, input)` that defines the forward pass. But it doesn't deal with any class data member. Instead, it's a pure function that takes both the model parameters and the inputs from the function parameters.

The following commands install dependencies and run the example:

```bash
python3 -m pip install --upgrade -r requirements.txt
python3 01-tutorial.py
```
