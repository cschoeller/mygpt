"""
Implementation of Muon Optimizer and Newton Schulz matrix orthogonalization.
Note: This file was partially AI generated.
"""

from collections.abc import Callable
import logging
from typing import Any

import torch
from torch.optim.optimizer import Optimizer, ParamsT

from optimizer.newton_schulz import NewtonSchulz

_logger = logging.getLogger(__name__)


class Muon(Optimizer):
    """Optimizer that applies a Newton Schulz iteration to the gradients before applying the update step.

    The definition is based on the Muon blogpost: https://kellerjordan.github.io/posts/muon/
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor,
        momentum: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        newton_schulz_iters: int = 5,
        *,
        nesterov: bool = True,
        maximize: bool = False,
        compile: bool = True,
    ) -> None:
        """Constructor of MuonOptimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            lr: Learning rate.
            momentum: EMA decay factor (beta) for the momentum buffer. 0.0 disables momentum.
            eps: Epsilon for numerical stability.
            weight_decay: AdamW-style decoupled weight decay.
            nesterov: Use Nesterov momentum. When True, the update blends the current gradient
                with the momentum buffer, giving the current gradient extra weight. This makes
                the optimizer more responsive to sudden gradient changes without accelerating
                the decay of the momentum buffer itself.
            maximize: Maximize the objective instead of minimizing.
            compile: Whether to compile the Newton-Schulz iteration for faster execution.
        """
        if lr < 0.0:
            msg = f"Invalid learning rate: {lr}"
            raise ValueError(msg)
        if isinstance(lr, torch.Tensor) and lr.ndim != 1:
            msg = f"Invalid learning rate tensor shape: {lr.shape}. Requires scalar or 1D tensor."
            raise ValueError(msg)
        if eps < 0.0:
            msg = f"Invalid epsilon value: {eps}"
            raise ValueError(msg)
        if not 0.0 <= momentum < 1.0:
            msg = f"Invalid momentum value: {momentum}. Must be in [0, 1)."
            raise ValueError(msg)

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "eps": eps,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "maximize": maximize,
        }
        super().__init__(params, defaults)
        self._compile = compile
        self._newton_schulz = torch.compile(
            NewtonSchulz(n=newton_schulz_iters, normalize=True), disable=not self._compile, mode="reduce-overhead"
        )

    @torch.no_grad()
    def step(  # typing: ignore[override]
        self,
        closure: Callable[[], float] | None = None,
    ) -> float | None:
        """Optimizer step that applies Newton Schulz orthogonalization to the gradients before applying the update step."""

        # iterate through each group
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wdc = group["weight_decay"]

            # iterate through each individual parameter in the group
            for p in group["params"]:
                if p.grad is None:
                    continue

                # flip sign if we maximize
                grad = -p.grad if group["maximize"] else p.grad

                # Apply momentum as an exponential moving average (EMA) of gradients:
                # v_t = beta * v_{t-1} + (1 - beta) * g_t
                if momentum > 0.0:
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    velocity = state["momentum_buffer"]
                    velocity.lerp_(grad, 1 - momentum)

                    # If Nesterov is applied, we add the gradient once more. This formulation
                    # comes from a reformulation via substitution of the original formula, and
                    # for practical reaons we assume g_t ~= g_t+1, which is especially true if we
                    # start with a small lr and warmup. Hence the formula is:
                    # update = beta*v_t + (1-beta)*g
                    if group["nesterov"]:
                        update = grad.lerp(velocity, momentum)
                    else:
                        update = velocity

                if grad.ndim >= 2:
                    grad_shape = grad.shape
                    update = update.view(grad_shape[0], -1)  # flatten all but the first dimension
                    update = self._newton_schulz(update)  # normalize and orthogonalize the gradient
                    update = update.view(grad_shape)  # restore original shape
                else:
                    self._warn_sgd_fallback_once(p, self.state[p])

                # AdamW-style weight decay
                p.mul_(1.0 - lr * wdc)

                # As NewtonSchulz computes an orthonormal approximation of grad, the update would
                # be stronger for matrices with cols > rows, than for rows > cols, because all columns
                # are unit length. To normalize this effect (its Frobenius norm), we scale the grad matrix
                # by sqrt(rows/cols).
                scale_factor = 1.0 if grad.ndim < 2 else max(1, grad.shape[0] / grad.shape[1]) ** 0.5

                # param update
                p.add_(-lr * scale_factor * update)

    def _warn_sgd_fallback_once(self, param: torch.Tensor, state: dict[str, Any]) -> None:
        """Log a warning the first time a parameter falls back to SGD due to insufficient dimensions."""
        if "sgd_fallback_logged" not in state:
            state["sgd_fallback_logged"] = True
            _logger.warning(
                "Parameter with shape %s has ndim < 2; falling back to SGD with momentum "
                "(Newton-Schulz orthogonalization skipped).",
                tuple(param.shape),
            )


def split_params_for_muon(
    model: torch.nn.Module,
    lm_head_names: tuple[str, ...] = ("lm_head", "_lm_head", "head", "output", "out_proj_head"),
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split parameters into Muon-compatible (hidden 2D+) and AdamW (everything else) groups.

    Following the Muon recommendations (Keller Jordan's blogpost and the Moonlight paper),
    only the *hidden* 2D+ weight matrices should be optimized with Muon. The following are
    explicitly routed to AdamW even when they have ndim >= 2:

    - All ``nn.Embedding`` parameters (including token and learned positional embeddings).
      Embedding rows are independent lookup vectors; orthogonalizing across the vocabulary
      dimension is semantically meaningless and empirically harmful.
    - The final language-model head (output projection to vocab logits). It benefits from
      AdamW's per-coordinate adaptive scaling and behaves more like an embedding than a
      hidden transform.

    1D/0D tensors (biases, LayerNorm gains, etc.) always go to AdamW since Newton-Schulz
    requires at least a matrix.

    Args:
        model: The model whose parameters should be split.
        lm_head_names: Substrings used to detect the LM head module by its fully qualified
            name. Match is performed against the last component of the dotted name so that
            e.g. ``transformer._lm_head`` is detected via ``"_lm_head"``.

    Returns:
        Tuple ``(muon_params, adamw_params)`` containing the parameters for the two
        optimizers. Every trainable parameter of the model appears in exactly one list.
    """
    # collect the set of parameter ids that should be excluded from Muon by name
    excluded_ids: set[int] = set()
    for module_name, module in model.named_modules():
        leaf_name = module_name.rsplit(".", 1)[-1]
        is_embedding = isinstance(module, torch.nn.Embedding)
        is_lm_head = leaf_name in lm_head_names
        if is_embedding or is_lm_head:
            for p in module.parameters(recurse=False):
                excluded_ids.add(id(p))

    # split parameters into Muon and AdamW groups based on dimensions and exclusion set
    muon_params: list[torch.nn.Parameter] = []
    other_params: list[torch.nn.Parameter] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and id(p) not in excluded_ids:
            muon_params.append(p)
        else:
            other_params.append(p)
    return muon_params, other_params
