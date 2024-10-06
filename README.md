# difusco

[![PyPI - Version](https://img.shields.io/pypi/v/difusco.svg)](https://pypi.org/project/difusco)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/difusco.svg)](https://pypi.org/project/difusco)

-----

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Dependencies

This project uses [hatch](https://hatch.pypa.io/) as a project manager. To install it, just `pip install hatch`. 

Unfortunately, two dependencies (`torch-scatter` and `torch-sparse`) require `torch` as an runtime dependency. The usual `hatch` dependency sync will not work for these two packages. To install them, do:

```bash
hatch run true # to create the environment and install basic dependencies

hatch shell # to enter the environment

pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.1+cpu.html
# pip install 'pyconcorde @ git+https://github.com/jvkersch/pyconcorde' -> currently not working

```
You only need to install `torch-scatter`, `torch-sparse` once. After that, you can use `hatch run` as usual to run the project, and dependencies will sync automatically (without removing the extra installed packages).

Generating the data for the TSP instances requires the `lkh` solver, which is run via the python wrapper `lkh`. To install it, use the [official LKH-3 site](
http://akira.ruc.dk/~keld/research/LKH-3/). Make sure to specify the ``--lkh_path`` argument pointing to the LKH-3 binary when generating the data with this solver.


## Data

We recommend saving the data in the following directory structure:

```bash
difusco/
├── data/                      # Directory for datasets
│   ├── tsp/                   # TSP dataset files
│   │   ├── tsp500_train_concorde.txt  # Training data for TSP
│   │   ├── tsp1000_train_concorde.txt # Additional training data for TSP
│   │   └── ...                # Other TSP data files
│   ├── problem_2/             # Data for Problem 2
│   │   ├── problem2_train.txt  # Training data for Problem 2
│   │   └── problem2_test.txt   # Test data for Problem 2
│   └── etc/                   # Other datasets or resources
│       ├── example_data.txt    # Example dataset
│       └── ...                # Other miscellaneous data files
```
### Traveling Salesman Problem

The data for the TSP comes from different sources. 
 - Files tsp{50,100}_{test,train}_concorde.txt come from [chaitjo/learning-tsp](https://github.com/chaitjo/learning-tsp) (Resources section).
 - Files tsp{500,1000,10000}_test_concorde.txt come from [Spider-scnu/TSP](https://github.com/Spider-scnu/TSP).
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


## License

`difusco` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
