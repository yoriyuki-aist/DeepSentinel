from chainer import links as L

from .utils import reshape_batch


class BatchBiLinear(L.Bilinear):

    def forward(self, e1, e2, n_batch_axes: int = 1):
        if n_batch_axes > 1:
            e1 = reshape_batch(e1, n_batch_axes)
            e2 = reshape_batch(e2, n_batch_axes)
        return super(BatchBiLinear, self).forward(e1, e2)
