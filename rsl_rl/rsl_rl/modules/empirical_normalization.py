import torch
import torch.nn as nn


class EmpiricalNormalization(nn.Module):
    """Running mean and variance normalizer.

    Maintains empirical estimates of the mean and variance of incoming data
    and uses them to normalize inputs. Statistics are updated only while the
    module is in training mode and until a specified number of samples has
    been observed.
    """

    def __init__(self, shape, until: float = 1.0e6, eps: float = 1.0e-8):
        super().__init__()
        self.shape = torch.Size(shape)
        self.until = until
        self.eps = eps
        self.register_buffer("count", torch.zeros(1))
        self.register_buffer("mean", torch.zeros(self.shape))
        self.register_buffer("var", torch.ones(self.shape))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update running statistics with a batch of samples."""
        batch_count = x.shape[0]
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = M2 / total_count
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count += batch_count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.count < self.until:
            self.update(x)
        return (x - self.mean) / torch.sqrt(self.var + self.eps)