import torch
import math


class CausalMath:
    """Causal inference utilities for backdoor adjustment and related operations."""

    @staticmethod
    def backdoor_adjustment(p_y_given_x_z: torch.Tensor, p_z: torch.Tensor, z_dim: int = -1) -> torch.Tensor:
        """Compute P(Y | do(X)) using the backdoor adjustment formula.

        P(Y | do(X)) = \\sum_z P(Y | X, Z=z) * P(Z=z)

        Args:
            p_y_given_x_z: Tensor representing P(Y | X, Z). Expected to have a
                dedicated dimension for Z (default: last dim).
            p_z: Tensor representing P(Z). Should sum to 1 over its last dimension.
            z_dim: The dimension index in `p_y_given_x_z` that corresponds to Z.

        Returns:
            Tensor representing P(Y | do(X)) with the Z dimension reduced.
        """
        # Normalize P(Z) in case it is not exactly normalized:
        p_z = p_z / p_z.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        # Align p_z shape for broadcasting into p_y_given_x_z
        target_shape = [1] * p_y_given_x_z.ndim
        target_shape[z_dim] = -1
        p_z_expanded = p_z.view(*target_shape)

        return (p_y_given_x_z * p_z_expanded).sum(dim=z_dim)


class LaplaceMath:
    """Manual mathematical development for Laplace Optimization [2, 5]."""
    
    @staticmethod
    def compute_fisher_block(input_batch: torch.Tensor, grad_output: torch.Tensor):
        """Computes KFAC-style Fisher blocks for a linear layer.

        Uses the approximation:
            F ≈ (A ⊗ G)
        where A = E[a a^T] and G = E[g g^T].

        Args:
            input_batch: activations before a linear layer (shape [N, d_in])
            grad_output: gradient w.r.t. layer output (shape [N, d_out])

        Returns:
            Tuple of (A, G) matrices, where A is [d_in, d_in] and G is [d_out, d_out].
        """
        n = input_batch.shape[0]
        if n == 0:
            raise ValueError("input_batch must have non-zero batch dimension")

        # Compute empirical second moments (scaled by 1/n)
        A = (input_batch.t() @ input_batch) / float(n)
        G = (grad_output.t() @ grad_output) / float(n)
        return A, G

    @staticmethod
    def calculate_log_det(precision_matrix: torch.Tensor):
        """Uses Cholesky decomposition for stable log-determinant calculation ."""
        return 2 * torch.log(torch.linalg.cholesky(precision_matrix).diagonal()).sum()

    @staticmethod
    def model_evidence(log_likelihood: float, precision_diag: torch.Tensor, n_params: int):
        """Computes the Laplace approximation to the marginal likelihood ."""
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

    @staticmethod
    def calculate_log_det(precision_matrix: torch.Tensor):
        """Uses Cholesky decomposition for stable log-determinant calculation ."""
        return 2 * torch.log(torch.linalg.cholesky(precision_matrix).diagonal()).sum()

    @staticmethod
    def model_evidence(log_likelihood: float, precision_diag: torch.Tensor, n_params: int):
        """Computes the Laplace approximation to the marginal likelihood ."""
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