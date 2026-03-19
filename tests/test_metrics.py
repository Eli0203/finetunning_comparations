import torch

from src.utils.metrics import natural_indirect_effect


def test_natural_indirect_effect_simple():
    # Setup a simple mediator distribution and outcome model.
    # Let P(Y|X=1,M) = [0.2, 0.8] for M=[0,1]
    p_y_given_x_m = torch.tensor([[0.2, 0.8]])

    # Let mediator distributions under X=0 and X=1
    p_m_x = torch.tensor([0.6, 0.4])
    p_m_x_prime = torch.tensor([0.2, 0.8])

    # NIE = E[Y_{x,M_{x'}}] - E[Y_{x,M_x}]
    # E[Y_{x,M_{x'}}] = 0.2*0.2 + 0.8*0.8 = 0.68
    # E[Y_{x,M_x}] = 0.2*0.6 + 0.8*0.4 = 0.44
    expected_nie = 0.68 - 0.44

    nie = natural_indirect_effect(p_y_given_x_m, p_m_x, p_m_x_prime, m_dim=1)
    assert torch.allclose(nie, torch.tensor([expected_nie]), atol=1e-6)
