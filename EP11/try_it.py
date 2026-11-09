# EP11 · Quantization in 15 lines: store each weight as a small integer + a shared scale
import torch
torch.manual_seed(0)
w = torch.randn(4096) * 0.08            # 4,096 weights, typical size about 0.05
w[7] = 2.8                              # one outlier, like the ones inside GPT-2
def int4(x):                            # 4 bits = 16 levels; we use 15 (-7 ... 7) so zero sits in the middle
    scale = x.abs().max() / 7           # the biggest weight sets the step
    q = (x / scale).round().clamp(-7, 7)    # the small integers you store
    return q * scale                    # what the model reads back
one = int4(w)                                     # ONE scale for everything
groups = torch.cat([int4(g) for g in w.split(32)])  # one scale per 32 weights
for name, r in [("one scale", one), ("groups of 32", groups)]:
    print(f"{name:13s} avg error {(r - w).abs().mean():.4f} | became zero: {100 * (r == 0).float().mean():.1f}%")
