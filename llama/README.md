This project depends on the Flax-based implementation of LLaMA from [EasyLM](https://github.com/young-geng/EasyLM).  Follow [these steps](https://github.com/young-geng/EasyLM#installation) to install EasyLM.

EasyLM depends on [mlxu](https://github.com/young-geng/mlxu). Follow [these steps](https://github.com/young-geng/mlxu#installation) to install mlxu.

Run the following command to git-clone the OpenLLaMA **3B** parameter checkpoint file

```bash
cd ~/work
git clone git@hf.co:openlm-research/open_llama_3b_600bt_preview_easylm
```

Run the following command to git-clone the OpenLLaMA **7B** parameter checkpoint file

```bash
cd ~/work
git clone git@hf.co:openlm-research/open_llama_7b_700bt_preview_easylm
```

Run the following command to load the **3B** checkpoint and generates the MLIR bytecode file `./llama3b-metal.mlir`.

```bash
cd flaxiree/llama
python3 export_llama
```

Run the following command to load the **7B** checkpoint and generates the MLIR bytecode file `./llama7b-metal.mlir`.

```bash
cd flaxiree/llama
python3 export_llama --config=7b
```
