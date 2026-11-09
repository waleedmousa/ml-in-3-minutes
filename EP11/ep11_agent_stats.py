"""EP11 · Outliers in our support bot (agent_ep10.pt from make_agent.py).
How big is the biggest weight compared with a typical one, and how many weights round to zero at 4 bits
with one scale for the whole table vs one scale per group of 32? Plus one weight, step by step, and 7B/70B sizes."""
import torch
SD = torch.load("agent_ep10.pt")
W = SD["p.transformer_h_0_attn_c_attn_weight"]                      # layer 1 attention table
mx, med = W.abs().max().item(), W.abs().median().item()
print(f"agent layer-1 attention matrix: {W.numel():,} weights, biggest |w| {mx:.3f} (value {W.flatten()[W.abs().argmax()].item():.3f}), median |w| {med:.4f}, ratio {mx/med:.1f}x")
s = mx / 7                                                           # 4-bit symmetric: values -7..7
print(f"  INT4 one scale: step {s:.4f}, rounded to zero {(torch.round(W / s) == 0).float().mean().item()*100:.1f}%")
g = W.reshape(-1, 32); sg = g.abs().amax(1, keepdim=True) / 7
print(f"  INT4 groups of 32: rounded to zero {(torch.round(g / sg) == 0).float().mean().item()*100:.1f}%")
M = SD["p.transformer_h_0_mlp_c_fc_weight"]; w = M[0, 0].item(); st = M.reshape(-1, 32)[0].abs().max().item() / 7
q = round(w / st)
print(f"one agent weight (layer 1 MLP [0,0]): {w:.6f} | 4-bit group-of-32 step {st:.6f} -> stored {q} -> read back {q*st:.6f} | error {q*st-w:+.6f}")
for n, name in ((7e9, "7B"), (70e9, "70B")):
    print(f"{name}: 16-bit {n*2/1e9:.0f} GB | 8-bit {n/1e9:.0f} GB | 4-bit (4.5 bits incl. scales) {n*4.5/8/1e9:.1f} GB")
