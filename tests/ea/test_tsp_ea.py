import os

from ea.tsp import TSPInstance, create_tsp_instance
from scipy.spatial import distance_matrix

from difusco.tsp.tsp_graph_dataset import TSPGraphDataset
from difusco.tsp.utils import TSPEvaluator


def get_tsp_sample() -> TSPInstance:
    resource_dir = "tests/resources"
    dataset = TSPGraphDataset(
        data_file=os.path.join(resource_dir, "tsp50_example_dataset_two_samples.txt"), sparse_factor=-1
    )

    return dataset.__getitem__(0)


def test_create_tsp_instance() -> None:
    sample = get_tsp_sample()
    instance = create_tsp_instance(sample, sparse_factor=-1)

    assert instance.gt_cost > 0
    assert instance.points.shape == (50, 2)
    assert instance.n == 50
    assert instance.gt_tour.shape == (51,)
    assert instance.gt_tour.min() == 0
    assert instance.gt_tour.max() == 49

    # Compare with hand-calculated cost
    tour = instance.gt_tour.cpu().numpy()
    points = sample[1].cpu().numpy()
    dist_mat = distance_matrix(points, points)

    calc_cost = 0
    for i in range(len(tour) - 1):
        calc_cost += dist_mat[tour[i], tour[i + 1]]

    assert abs(calc_cost - instance.gt_cost) < 1e-4

    # Compare with TSPEvaluator
    tsp_evaluator = TSPEvaluator(points)

    assert abs(tsp_evaluator.evaluate(tour) - instance.gt_cost) < 1e-4


# def test_tsp_ga_runs() -> None:
#     instance = read_tsp_instance()

#     ga = create_tsp_ea(instance, config=Config(pop_size=10))
#     ga.run(num_generations=2)

#     status = ga.status
#     assert status["iter"] == 2


# def test_tsp_problem_evaluation() -> None:
#     instance = get_tsp_sample()

#     problem = Problem(
#         objective_func=instance.evaluate_tsp_individual,
#         objective_sense="min",
#         solution_length=instance.n_nodes,
#         device="cpu",
#     )

#     # create ind as a random tensor with size n_nodes
#     ind = torch.rand(instance.n_nodes)
#     obj = instance.evaluate_tsp_individual(ind)
#     assert obj == problem._objective_func(ind)

#     # evaluate the ground truth tour of the instance
#     obj_gt = instance.evaluate_tsp_individual(instance.gt_tour)
#     assert obj_gt == problem._objective_func(instance.gt_tour)

#     assert obj_gt == instance.gt_tour.sum()
#     assert obj_gt <= obj
