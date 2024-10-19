from difusco.tsp.tsp_graph_dataset import TSPGraphDataset

# from difusco_edward_sun.difusco.co_datasets.tsp_graph_dataset import TSPGraphDataset


def test_tsp_dataset_is_loaded() -> None:
    dataset = TSPGraphDataset(
        data_file="tests/resources/tsp50_example_dataset.txt",
        sparse_factor=0.5,
    )
    assert len(dataset) == 30


def test_tsp_dataset_sample() -> None:
    dataset = TSPGraphDataset(
        data_file="tests/resources/tsp50_example_dataset.txt",
        sparse_factor=0,
    )

    sample = dataset.__getitem__(0)

    assert len(sample) == 4
    assert sample[1].shape == (50, 2)

    adj_matrix = sample[2]
    assert adj_matrix.shape == (50, 50)

    tour = sample[3]
    assert tour.shape == (50 + 1,)

    for i in range(tour.shape[0] - 1):
        assert tour[i] != tour[i + 1]
        assert adj_matrix[tour[i], tour[i + 1]] == 1


if __name__ == "__main__":
    test_tsp_dataset_is_loaded()
    test_tsp_dataset_sample()
