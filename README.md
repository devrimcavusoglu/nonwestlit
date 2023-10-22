
<p align="center">
<img src="https://site.unibo.it/nonwestlit/en/project/@@images/c716bd81-0be6-4d23-912f-01165f0391ee.jpeg">
<br>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/python-3.10-blue"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

<h1 align="center">NONWESTLIT</h1>


Project codebase for NONWESTLIT Project. The current primary focus within the project scope is to get experiment results on the dataset for text classification (3-way).

We aim to try:

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

Also, add the project root directory to PYTHONPATH to develop more smoothly without experiencing any path related problems. For earlier `conda` versions this was possible by `conda develop <project root dir>`, but it is [deprecated](https://github.com/conda/conda-build/issues/4251) (idk why?), so you can choose to manually add it to the PYTHONPATH. Heads up, this may require installing `conda-build` with miniconda.

## Usage

You can start training by the following command at the project root.

```shell
python nonwestlit train --model-name-or-path gpt2 --data-path test_data/toy_dataset.json --device cpu --output-dir outputs --bf16 1
```

The help docs/docstrings are not there yet, it will soon be ready.

For prediction

```shell
python nonwestlit predict --model-name-or-path gpt2 --device cpu
```

GPT-2 is used for test purposes, so no harm trying the commands above (you will need for the tests anyway). GPT-2 used for high-level functionality tests (training and prediction).

**Deepspeed**

It's currently experimental, and it's not tested yet. You can only use w/ GPU's having Ampere arch. (RTX 3000 series+ or workstation GPUs). I didn't perform training with it, but it seems promising.

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
