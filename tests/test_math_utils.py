import torch

from src.utils.math_utils import CausalMath, LaplaceMath


def test_backdoor_adjustment_basic():
    # p_y_given_x_z shape: [x=1, z=2]
    p_y_given_x_z = torch.tensor([[0.8, 0.2]])
    # P(Z) should be a distribution over z
    p_z = torch.tensor([0.25, 0.75])

    # Expected: 0.8*0.25 + 0.2*0.75 = 0.35
    result = CausalMath.backdoor_adjustment(p_y_given_x_z, p_z, z_dim=1)
    assert torch.allclose(result, torch.tensor([0.35]), atol=1e-6)


def test_compute_fisher_block_shapes():
    # Small batch (N=2), input dim=2, output dim=2
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    g = torch.tensor([[0.5, -0.5], [1.0, -1.0]])

    A, G = LaplaceMath.compute_fisher_block(x, g)

    # A should be 2x2 and equal to (x^T x)/N
    expected_A = (x.t() @ x) / x.shape[0]
    expected_G = (g.t() @ g) / g.shape[0]

    assert torch.allclose(A, expected_A)
    assert torch.allclose(G, expected_G)
