<p align="center">
<img src="https://site.unibo.it/nonwestlit/en/project/@@images/308d566c-4b9d-42c1-bcb6-d883a4057989.png">
<br>
<a href="https://www.python.org/downloads/release/python-31011/"><img alt="Code style: black" src="https://img.shields.io/badge/python-3.10-blue"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://arxiv.org/abs/2407.15136"><img alt="Code style: black" src="https://img.shields.io/badge/arXiv-2407.15136-b31b15.svg"></a>
</p>

<h1 align="center">NONWESTLIT</h1>


Project codebase for the paper [A multi-level multi-label text classification dataset of 19th century Ottoman and Russian literary and critical texts](https://arxiv.org/abs/2407.15136).

The objectives:

* Linear probing to the SOTA LLMs (e.g. Llama-2, Falcon).
* Fine-tune adapters e.g. LoRA.

## Setup

For the project environment and dependency management we use conda. If you do not have conda, I recommend installing `miniconda` as opposed to `anaconda` as the latter is shipped with redundant packages most of which we don't use at all. The initial setup can be conducted as follows. 

```shell
git clone git@github.com:devrimcavusoglu/nonwestlit.git
cd nonwestlit
conda env create -f environment.yml
```

Activate the created conda environment by `conda activate nonwestlit`. 

The following packages must be installed too for the integer quantization support (4bit & 8bit).

```shell
pip install bitsandbytes>=0.41.2.post2  # For integer quantization support
pip install peft==0.7.1  # For LoRA adapters
```

Also, add the project root directory to PYTHONPATH to develop more smoothly without experiencing any path related problems. For earlier `conda` versions this was possible by `conda develop <project root dir>`, but it is [deprecated](https://github.com/conda/conda-build/issues/4251) (idk why?), so you can choose to manually add it to the PYTHONPATH. Heads up, this may require installing `conda-build` with miniconda.

An alternative way to add the project root to the PYTHONPATH permanently for the environment, try the following:

1. Go to your conda dist. path (e.g. anaconda3, miniconda3, mamba) usually located at `$HOME/anaconda3`, from now on this path is referred as `CONDA_ROOT`
2. Find the activation bash file located at `CONDA_ROOT/envs/nonwestlit/etc/conda/activate.d`, the file name under this is `libxml2_activate.sh` for conda version `4.14.0` (could be different for future versions).
3. Open the file and add the following line and save and close it.
    ```shell
    export PYTHONPATH="${PYTHONPATH}:/path/to/project/root"
    ```
4. Deactivate and reactivate the environment `nonwestlit`. The project root is now permanently added to PYTHONPATH.

## Usage

To access the CLI and get information about available commands run the following:

```shell
python nonwestlit --help
# or python nonwestlit <command> --help
```

❗**Important Note:** The terminal commands given below are only for demonstration purposes, and may not represent all 
capability of the train arguments. The entry point `nonwestlit train` seamlessly support all HF TrainingArguments, 
just pass by the exact name and correct value. Please refer to [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)
to see all supported arguments for training.

You can start training by the following command at the project root. The following example training command was used 
for training [nonwestlit/falcon-7b-lora-seq-cls](https://huggingface.co/nonwestlit/falcon-7b-lora-seq-cls).

```shell
python nonwestlit train --model-name-or-path tiiuae/falcon-7b --train-data-path data/data-json/train.json --eval-data-path data/data-json/val.json --output-dir outputs/falcon_7b_lora_seq_cls --adapter lora --lora-target-modules ["query_key_value"] --bnb-quantization 4bit --experiment-tracking 1 --num-labels 3 --save-strategy steps --save-steps 0.141 --save-total-limit 1 --per-device-train-batch-size 2 --weight-decay 0.1 --learning-rate 0.00003 --eval-steps 0.047 --bf16 1 --logging-steps 1 --num-train-epochs 7
```

The help docs/docstrings are not there yet, it will soon be ready.

For prediction

```shell
python nonwestlit predict --model-name-or-path gpt2 --device cpu
```

GPT-2 is used for test purposes, so no harm trying the commands above (you will need for the tests anyway). GPT-2 used for high-level functionality tests (training and prediction).

**Deepspeed**

It's currently experimental, and it is not yet available in the main branch, and it's not tested yet. You can only use w/ GPU's having Ampere arch. (RTX 3000 series+ or workstation GPUs).

### Pushing the model to HF Hub

Use the following command to push your trained model to the huggingface-hub.

❗**Heads Up:** Before pushing the model to HF Hub you may want to remove the optimizer file saved along with the model, 
usually the optimizer binary files for big LLMs (e.g. llama-2, falcon) would be around 3 GB, and thus significantly 
slows down the uploading phase. Afaik, there is no way of certain ignoring files for huggingface-cli, so you have to 
either remove the optimizer file or move it outside the model directory.

```shell
  huggingface-cli upload nonwestlit LOCAL_MODEL_DIR HF_REPO_OR_MODEL_NAME --private --repo-type model
```

### Experiment Tracking

We are using Neptune.ai as an experiment tracking tool. To start logging neptune you need to create a simple config 
file as below, and name it `neptune.cfg` which must reside under the project root. Enter values without any quotation 
marks (shown in the example below).

```cfg
[credentials]
api_token=<YOUR_NEPTUNE_TOKEN>

[first-level-classification]
project=nonwestlit/first-level-classification

[second-level-classification]
project=nonwestlit/second-level-classification

# Append as necessary (if you create a new project)
[project-key]
project=<PROJECT_NAME>
```

Alternatively, you can set them as environment variables for project to be logged in set `NEPTUNE_PROJECT`, and for 
the API token set `NEPTUNE_API_TOKEN`. You can use `export` (MacOS, Linux) or `set` (Windows). For Linux you can 
set the environment variables as follows:

```shell
export NEPTUNE_PROJECT=<PROJECT_NAME> && export NEPTUNE_API_TOKEN=<YOUR_NEPTUNE_TOKEN>
```

Note that, since these are set for those sessions only, you need to set these environment variables once they expire
(e.g on reboot).

## Development

We mainly use `black` for the code formatting. Use the following command to format the codebase.

```shell
python -m scripts.run_code_style format
```

To check the code base is formatted, use

```shell
python -m scripts.run_code_style check
```

To run tests, use

```shell
python -m scripts.run_tests
```
