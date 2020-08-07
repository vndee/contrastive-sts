import torch


class AlignUniformLoss:
    def __init__(self, lam: float = 0.0):
        self.lam = lam

    def lalign(self, x, y, alpha=2):
        return (x - y).norm(dim=1).pow(alpha).mean()

    def lunif(self, x, t=2):
        sq_dist = torch.pdist(x, p=2).pow(2)
        return sq_dist.mul(-t).exp().mean().log()

    def __call__(self, x, y):
        """
        Calculate alignment and uniformity joint loss
        :param x: (batch_size, latent_dim)
        :param y: (batch_size, latent_dim)
        :return:
        """

        return self.lalign(x, y) + self.lam * (self.lunif(x) + self.lunif(y)) / 2
