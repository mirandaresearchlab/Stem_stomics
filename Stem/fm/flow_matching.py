import torch
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple


@dataclass
class FlowMatching:
    """
    Flow matching objective and sampler.

    The objective regresses the model output to the target velocity between
    data x0 and a Gaussian prior sample x1 along a linear interpolation path.
    Timesteps are sampled uniformly in [time_eps, 1 - time_eps] to avoid
    numerical issues at the endpoints.
    """

    time_eps: float = 1e-3
    time_scale: float = 1.0  # Multiply t before feeding to the model (keeps compat with discrete embedders).

    def _unwrap_model_output(self, model_out: torch.Tensor) -> torch.Tensor:
        """
        Convert model outputs to a single velocity field that matches x_t's shape.
        If the model returns two channels (mean + sigma) like the DDPM head,
        only the first channel is used as velocity.
        """
        if model_out.dim() < 3:
            raise ValueError(f"Expected model output with shape (B, C, ...), got {model_out.shape}")
        if model_out.shape[1] == 2:
            return model_out[:, :1]
        if model_out.shape[1] == 1:
            return model_out
        raise ValueError(
            f"Cannot infer velocity channel from shape {model_out.shape}; expected channel dim of 1 or 2."
        )

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # That 2* is to account for the fact that we want to sample in [time_eps, 1 - time_eps]
        return torch.rand(batch_size, device=device) * (1 - 2 * self.time_eps) + self.time_eps

    def training_losses(
        self,
        model: Callable[..., torch.Tensor],
        x0: torch.Tensor,
        model_kwargs: Optional[Dict] = None,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute per-sample flow matching losses.
        Args:
            model: callable like StemModel.forward(x, t, **model_kwargs).
            x0: data tensor of shape (B, C, ...).
            model_kwargs: conditioning dict passed through to the model.
            t: optional timesteps in [0, 1], shape (B,). If None, sampled uniformly.
            noise: optional prior sample x1. If None, draws standard normal.
        Returns:
            dict with key "loss" of shape (B,).
        """
        if model_kwargs is None:
            model_kwargs = {}
        if t is None:
            t = self.sample_time(x0.shape[0], x0.device)
        if noise is None:
            noise = torch.randn_like(x0)

        # chatGPT helped with the tensor shaping here
        t_scalar = t.reshape(-1)  # (B,)
        # expand t to broadcast over x0's shape (except batch dim)
        # if x0 has, say 200 dims, this makes t shape (B,1,1,1,...,1) with 200-1 ones
        # that leading * unpacks the list into separate arguments for view()
        t_broadcast = t_scalar.view(-1, *([1] * (x0.dim() - 1)))  # (B,1,...) for mixing x0/x1
        x1 = noise
        x_t = (1.0 - t_broadcast) * x0 + t_broadcast * x1
        target_v = x1 - x0

        model_out = model(x_t, t_scalar * self.time_scale, **model_kwargs)
        pred_v = self._unwrap_model_output(model_out)

        loss = ((pred_v - target_v) ** 2).flatten(1).mean(dim=1)
        return {"loss": loss}

    @torch.no_grad()
    def sample_euler(
        self,
        model: Callable[..., torch.Tensor],
        shape: Tuple[int, ...],
        num_steps: int,
        model_kwargs: Optional[Dict] = None,
        device: Optional[torch.device] = None,
        progress: bool = False,
    ) -> torch.Tensor:
        """
        Simple Euler integration of the learned velocity field from t=1 to t=0.
        Starts at Gaussian noise and steps along the flow: x_{t-dt} = x_t - dt * v.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if device is None:
            try:
                device = torch.device("gpu")
            except Exception:
                device = torch.device("cpu")

        x = torch.randn(*shape, device=device)
        dt = 1.0 / num_steps
        step_iter = range(num_steps)
        if progress:
            from tqdm.auto import tqdm

            step_iter = tqdm(step_iter)

        for i in step_iter:
            t_scalar = 1.0 - i * dt
            t_tensor = torch.full((shape[0],), t_scalar, device=device)
            v = self._unwrap_model_output(model(x, t_tensor * self.time_scale, **model_kwargs))
            x = x - dt * v
        return x


def create_flow_matching(time_eps: float = 1e-3, time_scale: float = 1.0) -> FlowMatching:
    """
    Factory to mirror create_diffusion.
    """
    return FlowMatching(time_eps=time_eps, time_scale=time_scale)
