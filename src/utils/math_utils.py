import torch
import math

class LaplaceMath:
    """Manual mathematical development for Laplace Optimization [2, 5]."""
    
    @staticmethod
    def compute_fisher_block(input_batch: torch.Tensor, grad_output: torch.Tensor):
        """
        Computes KFAC-style Fisher blocks for a linear layer [7, 9].
        F = (aa^T) ⊗ (gg^T) where a is input and g is output gradient.
        """
        # Outer product of inputs
        a_sq = torch.matmul(input_batch.t(), input_batch) / input_batch.shape
        # Outer product of gradients
        g_sq = torch.matmul(grad_output.t(), grad_output) / grad_output.shape
        return a_sq, g_sq

    @staticmethod
    def calculate_log_det(precision_matrix: torch.Tensor):
        """Uses Cholesky decomposition for stable log-determinant calculation [10, 11]."""
        return 2 * torch.log(torch.linalg.cholesky(precision_matrix).diagonal()).sum()

    @staticmethod
    def model_evidence(log_likelihood: float, precision_diag: torch.Tensor, n_params: int):
        """Computes the Laplace approximation to the marginal likelihood [10, 12]."""
        # Eq 14: P(y|X) ≈ exp(L(theta_MAP)) * (2pi)^(D/2) * |Sigma|^(1/2)
        log_det = torch.log(precision_diag).sum()
        evidence = log_likelihood - 0.5 * log_det + (n_params / 2) * math.log(2 * math.pi)
        return evidence
    
    @staticmethod
    def incremental_svd_update(current_B, new_activities, nkfac: int):
        """
        Algorithm 1: Efficient memory estimation of the low-rank factor B.
        Mantein BB^T ≈ Σ (g g^T) without computing the entire matrix.
        """
        # Combine current estimate with new gradients/activities
        B_prime = torch.cat([current_B, new_activities], dim=1)
        # SVD to maintain only the main components nkfac
        U, S, Vh = torch.linalg.svd(B_prime, full_matrices=False)
        return U[:, :nkfac] @ torch.diag(S[:nkfac])
    
    @staticmethod
    def log_det_kfac_low_rank(L_a, B_b, sigma_sq: float, n_lora: int, d: int):
        """
        E.2: Optimizing the evidence using the Determinant Lemma.
        Calculate log det(Σ_post^-1) eficiently[3].
        """
        # M = (I + σ^2 (L^T L ⊗ B^T B)) 
        # Reduce complexity  O(D^3) and O((nlora*nkfac)^3)
        LtL = L_a.t() @ L_a
        BtB = B_b.t() @ B_b
        
        # Kronecker small block product
        inner_matrix = torch.kron(LtL, BtB) * sigma_sq
        I_small = torch.eye(inner_matrix.size(0), device=inner_matrix.device)
        
        M = I_small + inner_matrix
        log_det_M = torch.logdet(M)
        
        # log det(precision) = -n_lora*d * log(σ^2) + log det(M) 
        return - (n_lora * d) * torch.log(torch.tensor(sigma_sq)) + log_det_M