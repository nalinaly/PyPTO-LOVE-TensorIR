import torch
import pypto.language as pl


@pl.jit
def kernel(x: pl.Tensor, w: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        for c in pl.range(w.shape[0] // 128):
            xt = pl.load(x, [0, 0], [1, x.shape[1]], target_memory=pl.MemorySpace.Mat)
            wt = pl.load(w, [c * 128, 0], [128, x.shape[1]], target_memory=pl.MemorySpace.Mat)
            mm = pl.gemv(xt, pl.tile.transpose_view(wt))
            y = pl.cast(mm, target_type=pl.BF16)
            pl.store(y, [0, c * 128], out)
    return out


def main():
    x = torch.empty((1, 256), dtype=torch.bfloat16, device="meta")
    w = torch.empty((256, 256), dtype=torch.bfloat16, device="meta")
    out = torch.empty((1, 256), dtype=torch.bfloat16, device="meta")
    print(kernel.specialize(x, w, out))


if __name__ == "__main__":
    main()
