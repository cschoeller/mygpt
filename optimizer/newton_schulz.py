"""Implementation of Newton Schulz matrix orthogonalization."""

import torch


class NewtonSchulz(torch.nn.Module):
    """Module that applies a Newton Schulz iteration for approximate matrix orthogonalization.

    Note that other coefficients, e.g., (2.0, -1.5, 0.5), can achieve a much more accurate orthogonalization, but
    require many more iterations, and are less robust.

    The definition is based on the Muon blogpost: https://kellerjordan.github.io/posts/muon/
    """

    def __init__(
        self,
        n: int = 5,
        coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        eps: float = 1e-7,
        *,
        normalize: bool = True,
    ) -> None:
        """Constructor of NewtonSchulz.

        Args:
            n: Number of iterations.
            coeffs: Coefficients for the polynomial
            eps: Epsilon for numerical stability when normalizing.
            normalize: Whether to normalize the input matrix before applying the iterations. Normalization is required
                for convergence guarantees.
        """
        super().__init__()
        assert len(coeffs) == 3, "Requires three coefficients."
        self._n = n
        self._a, self._b, self._c = coeffs
        self._eps = eps
        self._normalize = normalize

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run Newton Schulz for the number of defined iterations.

        The used odd polynomial we compute (more efficiently) is defined as:
        X' = aX + b(XX^T)*X + c(XX^T)**2*X
        """
        dim_a, dim_b = -2, -1
        size_a, size_b = x.size(dim_a), x.size(dim_b)

        if self._normalize:
            x = x / (x.norm() + self._eps)

        # leads to smaller gram matrix and must be undone at the end
        if size_a > size_b:
            x = x.transpose(dim_a, dim_b)

        for _ in range(self._n):
            x_a = x @ x.transpose(dim_a, dim_b)
            x_b = x_a @ x
            x = self._a * x + self._b * x_b + self._c * x_a @ x_b

        if size_a > size_b:
            x = x.transpose(dim_a, dim_b)

        return x

    @staticmethod
    def orth_error(x: torch.Tensor) -> float:
        """Helper function to compute the orthogonality error of a matrix for testing."""
        x_squared = (x.transpose(0, 1) @ x) - torch.eye(x.size(1))
        return x_squared.norm().item()
