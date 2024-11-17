# ruff: noqa: ANN201

import click
from ea.evolutionary_algorithm import main_ea

from difusco.difusco import main_difusco
from difusco.mis.generate_node_degree_labels import generate_node_degree_labels_main
from difusco.tsp.generate_tsp_data import main_tsp_data_generation
from difusco.tsp.run_tsp_heuristics import run_tsp_heuristics_main


@click.group()
def cli():
    """A CLI tool for the DDM - EA project."""


@click.group()
def difusco_group():
    """Commands related to Difusco."""


@difusco_group.command()
def run_difusco():
    """Run the Difusco main command."""
    main_difusco()


@difusco_group.command()
def generate_tsp_data():
    """Generate TSP data."""
    main_tsp_data_generation()


@difusco_group.command()
def generate_node_degree_labels():
    """Generate MIS degree labels."""
    generate_node_degree_labels_main()


@difusco_group.command()
def run_tsp_heuristics():
    """Run TSP heuristics."""
    run_tsp_heuristics_main()


@click.group()
def ea_group():
    """Commands related to the EA project."""


@ea_group.command()
def run_ea():
    """Run the Evolutionary Algorithm."""
    main_ea()


cli.add_command(difusco_group, name="difusco")
cli.add_command(ea_group, name="ea")

if __name__ == "__main__":
    cli()
