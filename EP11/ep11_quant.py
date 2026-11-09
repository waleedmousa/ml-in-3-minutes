"""EP11 · Quantization on the real GPT-2 (124M), plain round-to-nearest (no GPTQ/AWQ).
Weight matrices are rounded to 8, 4, 3 or 2 bits with ONE scale per matrix, one per column, or one per group of 32.
Biases and layer-norm stay FP32 (0.1% of the weights). Sizes count the stored scales.
We measure: size, average loss on 20 everyday sentences and on our 54 support chats, and GPT-2's next word."""
import sys, time, torch, tiktoken, torch.nn.functional as F
sys.path.insert(0, "."); import gpt2_torch
enc = tiktoken.get_encoding("gpt2"); m = gpt2_torch.load(); m.eval()
ORIG = {k: v.detach().clone() for k, v in m.p.items()}
MATS = [k for k, v in ORIG.items() if v.dim() == 2]
TEXT = [l.strip() for l in open("eval_text.txt") if l.strip()]
PROMPT = "I was charged twice, so I want a"
n_all = sum(v.numel() for v in ORIG.values()); n_mat = sum(ORIG[k].numel() for k in MATS)
print(f"weights: {n_all:,} | in matrices: {n_mat:,} ({100*n_mat/n_all:.2f}%)")

def quant(w, bits, group=None):
    qmax = 2 ** (bits - 1) - 1                       # 8-bit: -127..127 · 4-bit: -7..7
    if group is None: s = w.abs().max() / qmax       # one scale for the whole matrix
    else:
        g = w.reshape(-1, group); s = g.abs().amax(1, keepdim=True) / qmax
        return (torch.round(g / s).clamp(-qmax, qmax) * s).reshape(w.shape)
    return torch.round(w / s).clamp(-qmax, qmax) * s

@torch.no_grad()
def loss():
    tot = 0
    for t in TEXT:
        ids = torch.tensor(enc.encode(t)); tot += F.cross_entropy(m(ids)[:-1], ids[1:]).item()
    return tot / len(TEXT)

@torch.no_grad()
def nxt(k=3):
    ids = torch.tensor(enc.encode(PROMPT)); p = m(ids)[-1].softmax(-1); v, i = p.topk(k)
    return ", ".join(f"{enc.decode([int(j)]).strip()} {100*float(x):.1f}%" for x, j in zip(v, i))

@torch.no_grad()
def greedy(n=10):
    ids = enc.encode(PROMPT)
    for _ in range(n): ids.append(int(m(torch.tensor(ids))[-1].argmax()))
    return enc.decode(ids[len(enc.encode(PROMPT)):])

def quant_asym(w, bits, group):                     # zero-point version: each group stores its own min and step
    g = w.reshape(-1, group); lo = g.amin(1, keepdim=True); hi = g.amax(1, keepdim=True); L = 2 ** bits - 1
    s = (hi - lo) / L; z = torch.round(-lo / s); q = (torch.round(g / s) + z).clamp(0, L); return ((q - z) * s).reshape(w.shape)
def quant_col(w, bits):                             # one scale per output column (per-channel)
    qmax = 2 ** (bits - 1) - 1; s = w.abs().amax(0, keepdim=True) / qmax; return torch.round(w / s).clamp(-qmax, qmax) * s
BLOCK = [k for k in MATS if "_h_" in k]; EMB = [k for k in MATS if k not in BLOCK]
CH = [l.strip() for f in ("support_chats.txt", "support_validation.txt", "support_test_final.txt") for l in open(f) if l.strip()]
@torch.no_grad()
def loss_chats():
    t = 0
    for c in CH: ids = torch.tensor(enc.encode(c)); t += F.cross_entropy(m(ids)[:-1], ids[1:]).item()
    return t / len(CH)
def run(name, plan):                                # plan(key) -> (function, bits per weight incl. scales) or None
    with torch.no_grad():
        bytes_ = 0
        for k, v in ORIG.items():
            pl = plan(k) if k in MATS else None
            m.p[k].copy_(pl[0](v.clone()) if pl else v); bytes_ += v.numel() * (pl[1] / 8 if pl else 4)
    print(f"{name:38s} | {bytes_/1e6:6.1f} MB | text loss {loss():6.3f} | chat loss {loss_chats():6.3f} | next: {nxt()} | greedy: {greedy()!r}", flush=True)
run("FP32 original", lambda k: None)
run("FP16", lambda k: (lambda w: w.half().float(), 16))
run("INT8, one scale per column", lambda k: (lambda w: quant_col(w, 8), 8))
run("INT8, ONE scale per matrix", lambda k: (lambda w: quant(w, 8), 8))
run("INT4, ONE scale per matrix (naive)", lambda k: (lambda w: quant(w, 4), 4))
run("INT4, groups of 32, everything", lambda k: (lambda w: quant(w, 4, 32), 4.5))
run("INT4 blocks g32 + token table INT8", lambda k: (lambda w: quant_asym(w, 4, 32), 5) if k in BLOCK else (lambda w: quant_col(w, 8), 8))
run("INT3 blocks g32 + token table INT8", lambda k: (lambda w: quant_asym(w, 3, 32), 4) if k in BLOCK else (lambda w: quant_col(w, 8), 8))
run("INT2 blocks g32 + token table INT8", lambda k: (lambda w: quant_asym(w, 2, 32), 3) if k in BLOCK else (lambda w: quant_col(w, 8), 8))
run("ONLY the token table INT4 g32", lambda k: (lambda w: quant(w, 4, 32), 4.5) if "wte" in k else None)
run("ONLY the token table INT8 per column", lambda k: (lambda w: quant_col(w, 8), 8) if "wte" in k else None)
print("bits per weight include the stored scales: symmetric groups of 32 = 4 + 16/32 = 4.5; zero-point groups of 32 = 4 + 32/32 = 5")

# The outlier problem, on one real matrix
run('back to FP32', lambda k: None)
print("\n--- outliers ---")
for k in MATS:
    w = ORIG[k]; r = float(w.abs().max() / w.abs().median())
    if k.endswith("h_0_attn_c_attn_weight") or k.endswith("h_5_mlp_c_fc_weight") or k.endswith("wte_weight"):
        q = quant(w, 4); z = float((q == 0).float().mean()); qg = quant(w, 4, 32); zg = float((qg == 0).float().mean())
        print(f"{k}: biggest {float(w.abs().max()):.3f} vs typical(median |w|) {float(w.abs().median()):.4f} = {r:.0f}x | "
              f"INT4 one scale: step {float(w.abs().max()/7):.4f}, {100*z:.1f}% of weights become 0 | groups of 32: {100*zg:.1f}% become 0")
big = max(MATS, key=lambda k: float(ORIG[k].abs().max() / ORIG[k].abs().median()))
w = ORIG[big]; print(f"most extreme matrix: {big}: max {float(w.abs().max()):.3f}, median |w| {float(w.abs().median()):.4f}, ratio {float(w.abs().max()/w.abs().median()):.0f}x, "
                     f"INT4 one-scale zeros {100*float((quant(w,4)==0).float().mean()):.1f}%, groups-of-32 zeros {100*float((quant(w,4,32)==0).float().mean()):.1f}%")
# one real weight, rounded
w0 = ORIG["transformer_h_0_mlp_c_fc_weight"]; x = float(w0[0, 0]); s8 = float(w0.abs().max() / 127); s4 = float(w0.reshape(-1, 32)[0].abs().max() / 7)
print(f"\none weight (layer 1 MLP [0,0]): {x:.6f} | INT8 one scale: step {s8:.6f} -> stored {round(x/s8)} -> {round(x/s8)*s8:.6f} | INT4 group 32: step {s4:.6f} -> stored {round(x/s4)} -> {round(x/s4)*s4:.6f}")
