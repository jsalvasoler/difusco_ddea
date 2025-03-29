# Diffusion-Based Evolutionary Algorithms

This repository contains the code of my Master Thesis

-----

## Table of Contents

- [Dependencies](#dependencies)
- [Data](#data)
  - [Traveling Salesman Problem](#traveling-salesman-problem)
  - [Maximum Independent Set (MIS)](#maximum-independent-set-mis)
- [Models](#models)
- [License](#license)

## Dependencies and Installation

This project uses [hatch](https://hatch.pypa.io/) as a project manager. To install it, just `pip install hatch` or `pipx install hatch`.

Unfortunately, two dependencies (`torch-scatter` and `torch-sparse`) require `torch` as an runtime dependency. The usual `hatch` dependency sync will not work for these two packages. To install them, do:

```bash

hatch shell # to create and enter the environment

pip install torch-scatter torch-sparse -f https://pytorch-geometric.com/whl/torch-2.3.1+cu121.html

```
You only need to install `torch-scatter`, `torch-sparse` once. After that, you can use `hatch run` as usual to run the project, and dependencies will sync automatically (without removing the extra installed packages).

Generating the data for the TSP instances requires the `lkh` solver, which is run via the python wrapper `lkh`. To install it, use the [official LKH-3 site](
http://akira.ruc.dk/~keld/research/LKH-3/). Make sure to specify the ``--lkh_path`` argument pointing to the LKH-3 binary when generating the data with this solver.

Finally, we need to compile the `cython` code for the TSP heuristics. To do so, run the following command:

```bash
./src/problems/tsp/cython_merge/compile.sh
```


## CLI Usage

There is a simple click CLI that can be used to run all the relevant modules. To see the available commands, run:

```bash
hatch run cli --help
```

There are two groups of commands: `difusco` and `ea`. Run `hatch run cli difusco --help` and `hatch run cli ea --help` to see the available commands for each group.


## Testing

Set up the test environment by running:

```bash
hatch shell hatch-test.py3.11

pip install torch-scatter torch-sparse -f https://pytorch-geometric.com/whl/torch-2.3.1+cu121.html
```

Run the tests (in parallel if -p is specified) with:

```bash
hatch test -p
```

Some tests require the data to be present, otherwise they will be skipped. Same with cuda availability.



## Data

While path engineering allows for flexibility in script compatibility with various directory structures, we strongly recommend adhering to the following directory structure for optimal organization and ease of use:

```bash
data/                      # Directory for datasets
├── tsp/                   # TSP dataset files
│   ├── tsp500_train_concorde.txt  # Training data for TSP
│   ├── tsp1000_train_concorde.txt # Additional training data for TSP
│   └── ...                # Other TSP data files
├── mis/                   # MIS dataset files
│   ├── er_50_100/         # ER-50-100 dataset
│   │   ├── test           # Test instances
│   │   ├── test_labels    # Labels for test instances
│   │   ├── train          # Training instances
│   │   └── train_labels   # Labels for training instances
│   ├── er_700_800/        # ER-700-800 dataset
│   │   ├── test           # Test instances
│   │   ├── test_labels    # Labels for test instances
│   │   ├── train          # Training instances
│   │   └── train_labels   # Labels for training instances
│   ├── satlib/            # SATLIB dataset
│   │   ├── test           # Test instances
│   │   ├── test_labels    # Labels for test instances
│   │   ├── train          # Training instances
│   │   └── train_labels   # Labels for training instances
│   └── ...                # Other MIS data files
├── difuscombination/      # Directory for diffusion combination experiments
│   └── mis/               # MIS experiments for diffusion combination
│       ├── er_50_100/     # ER-50-100 dataset
│       │   ├── test       # Test instances directory
│       │   ├── test_labels# Labels for test instances directory
│       │   ├── train      # Training instances directory
│       │   └── train_labels# Labels for training instances directory
│       ├── er_300_400/    # ER-300-400 dataset
│       │   ├── test       # Test instances directory
│       │   ├── test_labels# Labels for test instances directory
│       │   ├── train      # Training instances directory
│       │   └── train_labels# Labels for training instances directory
│       └── er_700_800/    # ER-700-800 dataset
│           ├── test       # Test instances directory
│           ├── test_labels# Labels for test instances directory
│           ├── train      # Training instances directory
│           └── train_labels# Labels for training instances directory
└── etc/                   # Other datasets or resources
    ├── example_data.txt    # Example dataset
    └── ...                # Other miscellaneous data files
```

### Downloading Data

TODO: Add links to the data here.
TODO: how to unzip and set it up


### Traveling Salesman Problem (TSP)

The downloaded data contains the testing sets for the TSP instances. The origin of the training sets are the following:

The data for the TSP comes from different sources. 
 - Files tsp{50,100}_{test,train}_concorde.txt come from [chaitjo/learning-tsp](https://github.com/chaitjo/learning-tsp) (Resources section).
 - Files tsp{500,1000,10000}_test_concorde.txt come from [Spider-scnu/TSP](https://github.com/Spider-scnu/TSP) (Dataset section).
 - Files tsp{500,1000,10000}_train_concorde.txt are generated using the following commands:

```bash
hatch run difusco generate_tsp_data \
  --min_nodes 500 \
  --max_nodes 500 \
  --num_samples 128000 \
  --batch_size 128 \
  --filename "/data/tsp/tsp500_train_concorde.txt" \
  --seed 1234 \
  --lkh_path "/path/to/lkh"
```

```bash
hatch run difusco generate_tsp_data \
  --min_nodes 1000 \
  --max_nodes 1000 \
  --num_samples 64000 \
  --batch_size 128 \
  --filename "/data/tsp/tsp1000_train_concorde.txt" \
  --seed 1234 \
  --lkh_path "/path/to/lkh"
```

```bash
hatch run difusco generate_tsp_data \
  --min_nodes 10000 \
  --max_nodes 10000 \
  --num_samples 64000 \
  --batch_size 65 \
  --filename "/data/tsp/tsp10000_train_concorde.txt" \
  --seed 1234 \
  --lkh_path "/path/to/lkh"
```

### Maximum Independent Set (MIS)

Each MIS dataset directory (e.g., er_50_100, er_300_400, er_700_800, satlib) follows the same structure containing four files:

- `test`: Contains the test graph instances
- `test_labels`: Contains the optimal/best-known solution labels for test instances
- `train`: Contains the training graph instances
- `train_labels`: Contains the optimal/best-known solution labels for training instances


The downloaded data contains the testing sets for the MIS instances. The er_700_800 and satlib datasets come from [Difusco - Sun et al. (2023)](https://github.com/Edward-Sun/DIFUSCO). The training sets, and the rest of the test / train sets are generated using the following commands:

```bash
min_nodes=700
max_nodes=800
num_graphs=163840
er_p=0.15

python -u src/mis_benchmark_framework/main.py gendata \
    random \
    None \
    /your/path/to/data_er/train \
    --model er \
    --min_n $min_nodes \
    --max_n $max_nodes \
    --num_graphs $num_graphs \
    --er_p $er_p
```

## Models

We provide the pretrained models for the TSP (initialization) and MIS (initialization and recombination) in the following links:

TODO: add links to the models and how to download them

The TSP models come from the work of [Difusco - Sun et al. (2023)](https://github.com/Edward-Sun/DIFUSCO). The MIS models for er_700_800 and satlib as well. The rest of the models are trained by us.

We recommend saving the models in the following directory structure:

```bash
├── models/                    # Directory for models
│   ├── tsp/                   # TSP models
│   ├── mis/                   # MIS models
│   └── difuscombination/      # Diffusion recombination models
│       ├── tsp/               # TSP diffusion recombination models
│       └── mis/               # MIS diffusion recombination models
```

## License

This repository is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
