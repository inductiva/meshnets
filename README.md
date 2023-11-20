# Meshnets project

While virtual wind tunnels are a much cheaper and flexible option than
using a real physical wind tunnel for initial design iterations,
realistic and high-fidelity simulations still require massive amounts
of computation. Typical computational fluid dynamic (CFD) simulations,
like the ones shown above, may take many hours to run, even on
large-scale computational infrastructures. This hinders the ability of
engineers to explore more variations of the design and slows down the
speed of project development. This is where Machine Learning
techniques can help the world of simulation and engineering.

In this repository you will find the necessary code to train the
models on data obtained from virtual wind tunnels.

# Running the code

This section will go over the necessary steps to train your own
models.

## The model

The graph encoding and the model in this repository are inspired by the [MeshGraphNets](https://arxiv.org/abs/2010.03409) paper from DeepMind.
Notable differences in our implementation are: a single set of edges in the graph encoding, and the use of the model as a single-step predictor for a static physical property.

Note that the `torch` model can be used freely with any graph data and is not limited to the wind tunel scenario, or even mesh related tasks.

We suggest you to have a look at the paper if you are interested in knowing more details about the model and data encoding, it is a great read!

## Mlflow

We are using
[mlflow](https://mlflow.org/docs/latest/python_api/mlflow.html) to
keep track of our experiments. We have our own remote server to which
we can log everything from any computer which was created following
the instructions in [this
tutorial](https://towardsdatascience.com/managing-your-machine-learning-experiments-with-mlflow-1cd6ee21996e).

This is a nice feature to have but it is not required in any way to
run our code. Everything will be logged locally by default. That is,
all experiments will be logged to the folder `mlruns` created in the
directory from which the script is launched. To then look at the
experiments in the browser we just need to run the command `mlflow
ui`.

## Installing everything

The next step is to actually clone the repository using:

```bash
git clone https://github.com/inductiva/meshnets.git
```

The very next step is to create a virtual environment. This will solve
any clashes with the library versions used here and anything else that
might be installed in your own system:

```bash
python3 -m venv .env
source .env/bin/activate
```

After creating and activating the virtual environment we can install
all the requirements of the project using:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Next, because `meshnets` is actually packaged we can install it using:

```bash
pip install -e .
```

## Training a model

Now to the fun part. Training a model is simply a matter of running
the script `train.py`:

```python
python train.py --dataset_version=<dataset_version>
--val_data_dirs=<Path(s) to validation datasets> --other_flags
```

where `dataset_version` tells the script which dataset version from
hugging face to use.

all other flags, their description and default values can be inspected
by running:

```python
python train.py --help
```

**Note**: Depending on your machine you may need to set the `num_cpus_per_worker` to an appropriate number of CPUs. A number higher than your machine/cluster has and it will block training from being launched.

## Additional scripts

We encourage you to also make use of the other scripts depending on your needs.

Explore how hyperparmaters affect the results with `tune.py`:

```python
python tune.py --data_dir=<Path to your dataset>
--val_data_dirs=<Path(s) to validation datasets> --other_flags
```

Moreover, if you are logging your models with `mlflow`, you can easily load them to conduct further analysis.

Evaluate how well a model performs on a dataset by computing its loss with `eval.py`:

```python
python eval.py --data_dir=<Path to your dataset>
--run_id=<Run ID of the experiment to load> --checkpoint=<Checkpoint to load from the experiment> --other_flags
```

Plot a comparison between groundtruth and your model's prediction with `visualize.py`

```python
python visualize.py --data_dir=<Path to your dataset>
--run_id=<Run ID of the experiment to load> --checkpoint=<Checkpoint to load from the experiment> --other_flags
```
