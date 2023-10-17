import random
import ray
import numpy as np
import torch


def _assert_np_random():
    np_rand = np.random.random()
    assert np.isclose(np_rand, 0.417022004702574)


def _assert_py_random():
    rand = random.random()
    assert np.isclose(rand, 0.13436424411240122)


def _assert_torch_random():
    rand_torch = torch.rand(1).item()
    assert np.isclose(rand_torch, 0.7576315999031067)


def test_conftest():
    _assert_np_random()
    _assert_py_random()
    _assert_torch_random()


def test_conftest_order():
    _assert_torch_random()
    _assert_py_random()
    _assert_np_random()


def test_ray_cluster(ray_cluster):
    ray_cluster.setUp()
    assert len(ray_cluster.node_ips()) == ray_cluster.nodes
    ray.shutdown()
    ray_cluster.setUp()
    assert len(ray_cluster.node_ips()) == ray_cluster.nodes
