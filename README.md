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
```
You only need to install `torch-scatter` and `torch-sparse` once. After that, you can use `hatch run` as usual to run the project, and dependencies will sync automatically (without removing the two packages).

## License

`difusco` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
