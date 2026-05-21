"""
Vanilla conditional VAE (CVAE) for spot-level gene expression prediction.

This is the VAE counterpart of the Stem (diffusion) and FM (flow matching)
models: it models p(X | V) where X is the log2-transformed gene expression
vector over a fixed gene panel and V is the paired histology patch, represented
by the same UNI+CONCH image embedding the DiT models consume.

Deliberately a plain MLP (no DiT backbone): with a fixed gene panel the input
is a fixed-dim vector, so the per-gene token structure of the DiT buys nothing
here. The architecture mismatch with Stem/FM is an accepted, disclosed choice;
total parameter count is matched in the "large" config.
"""

import math

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: int, out_dim: int, num_layers: int) -> nn.Sequential:
    """MLP with `num_layers` SiLU-activated layers + a linear output head.

    num_layers=1 -> Linear(in,hidden), SiLU, Linear(hidden,out).
    Each extra layer adds one hidden Linear(hidden,hidden)+SiLU.
    """
    assert num_layers >= 1, "num_layers must be >= 1"
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.SiLU()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class ConditionalVAE(nn.Module):
    def __init__(
        self,
        input_size: int = 200,
        label_size: int = 1536,
        hidden_size: int = 2304,
        num_layers: int = 4,
        latent_dim: int = 256,
        cond_hidden_size: int = 512,
    ):
        """
        input_size:       number of genes in the panel (C)
        label_size:       raw image-embedding dim (UNI+CONCH concat, same as Stem/FM)
        hidden_size:      width of the encoder/decoder MLPs
        num_layers:       SiLU-activated layers per encoder/decoder MLP
        latent_dim:       VAE latent dimension
        cond_hidden_size: width of the conditioning projection
        """
        super().__init__()
        self.input_size = input_size
        self.label_size = label_size
        self.latent_dim = latent_dim

        # project the raw UNI+CONCH embedding (the exact input Stem/FM condition on)
        self.cond_proj = nn.Sequential(
            nn.Linear(label_size, cond_hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(cond_hidden_size, cond_hidden_size, bias=True),
        )
        # encoder q(z | X, V): [X ; c] -> (mu, logvar)
        self.encoder = _mlp(input_size + cond_hidden_size, hidden_size, 2 * latent_dim, num_layers)
        # decoder p(X | z, V): [z ; c] -> X_hat
        self.decoder = _mlp(latent_dim + cond_hidden_size, hidden_size, input_size, num_layers)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def encode(self, x: torch.Tensor, c: torch.Tensor):
        h = self.encoder(torch.cat([x, c], dim=-1))
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, c], dim=-1))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        x: (N, NumGene) clean log2 gene expression
        y: (N, label_size) raw image embedding
        returns: x_recon (N, NumGene), mu (N, latent_dim), logvar (N, latent_dim)
        """
        c = self.cond_proj(y)
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        return x_recon, mu, logvar

    @torch.no_grad()
    def sample(self, y: torch.Tensor) -> torch.Tensor:
        """Draw one sample per conditioning row: z ~ N(0, I), decode. -> (N, NumGene)."""
        c = self.cond_proj(y)
        z = torch.randn(y.shape[0], self.latent_dim, device=y.device, dtype=c.dtype)
        return self.decode(z, c)


def cvae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> dict:
    """beta-weighted negative ELBO. Reconstruction is a fixed-variance Gaussian NLL
    (= MSE), summed over genes; KL is summed over the latent dim; both mean-reduced
    over the batch. Matches the log2-expression space used everywhere else."""
    recon = ((x - x_recon) ** 2).sum(dim=-1).mean()
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
    loss = recon + beta * kl
    return {"loss": loss, "recon": recon, "kl": kl}


def CVAE(**kwargs) -> ConditionalVAE:
    return ConditionalVAE(**kwargs)


CVAE_models = {"CVAE": CVAE}
