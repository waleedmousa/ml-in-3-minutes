"""EP11 · Shrinking our EP10 support agent (GPT-2 124M fine-tuned on 40 support chats) before we ship it.
Same rounding functions as ep11_quant.py (plain round-to-nearest). For each version we check:
size, test loss on the 6 test chats, and its replies to the 14 chats it never trained on (8 validation + 6 test)."""
import sys, torch, tiktoken, torch.nn.functional as F
sys.path.insert(0, "."); import gpt2_torch
enc = tiktoken.get_encoding("gpt2"); EOT = enc.eot_token
m = gpt2_torch.load(); m.eval(); SD = torch.load("agent_ep10.pt"); m.load_state_dict(SD)
ORIG = {k: v.detach().clone() for k, v in m.p.items()}; MATS = [k for k, v in ORIG.items() if v.dim() == 2]
BLOCK = [k for k in MATS if "_h_" in k]
def quant(w, bits, group=None):
    qmax = 2 ** (bits - 1) - 1
    if group is None: s = w.abs().max() / qmax; return torch.round(w / s).clamp(-qmax, qmax) * s
    g = w.reshape(-1, group); s = g.abs().amax(1, keepdim=True) / qmax
    return (torch.round(g / s).clamp(-qmax, qmax) * s).reshape(w.shape)
def quant_asym(w, bits, group):
    g = w.reshape(-1, group); lo = g.amin(1, keepdim=True); hi = g.amax(1, keepdim=True); L = 2 ** bits - 1
    s = (hi - lo) / L; z = torch.round(-lo / s); q = (torch.round(g / s) + z).clamp(0, L); return ((q - z) * s).reshape(w.shape)
def quant_col(w, bits):
    qmax = 2 ** (bits - 1) - 1; s = w.abs().amax(0, keepdim=True) / qmax; return torch.round(w / s).clamp(-qmax, qmax) * s
VAL = [l.strip() for l in open("support_validation.txt") if l.strip()]; TEST = [l.strip() for l in open("support_test_final.txt") if l.strip()]
HO = VAL + TEST
@torch.no_grad()
def test_loss():
    t = 0
    for c in TEST: ids = torch.tensor(enc.encode(c) + [EOT]); t += F.cross_entropy(m(ids)[:-1], ids[1:]).item()
    return t / len(TEST)
@torch.no_grad()
def reply(q, n=30):
    ids = enc.encode(f"{q} Agent:"); N = len(ids)
    for _ in range(n):
        x = int(m(torch.tensor(ids))[-1].argmax())
        if x == EOT: break
        ids.append(x)
    return enc.decode(ids[N:]).strip()
BASE = None
def run(name, plan):
    global BASE
    with torch.no_grad():
        b = 0
        for k, v in ORIG.items():
            pl = plan(k) if k in MATS else None
            m.p[k].copy_(pl[0](v.clone()) if pl else v); b += v.numel() * (pl[1] / 8 if pl else 4)
    reps = [reply(c.split(" Agent:")[0]) for c in HO]
    if BASE is None: BASE = reps
    same = sum(a == r for a, r in zip(BASE, reps)); right = sum(r == c.split(" Agent:")[1].strip() for r, c in zip(reps, HO))
    print(f"{name:40s} | {b/1e6:6.1f} MB | test loss {test_loss():6.3f} | same reply as original {same:2d}/14 | exact reference reply {right:2d}/14", flush=True)
    print(f"   'Where is my refund?' -> {reps[3]!r}")
    print(f"   'I lost my debit card.'-style ('I left my card in a taxi.') -> {reps[9]!r}", flush=True)
run("FP32 original agent", lambda k: None)
run("FP16", lambda k: (lambda w: w.half().float(), 16))
run("INT8, one scale per column", lambda k: (lambda w: quant_col(w, 8), 8))
run("INT8, ONE scale per matrix", lambda k: (lambda w: quant(w, 8), 8))
run("INT4, ONE scale per matrix (naive)", lambda k: (lambda w: quant(w, 4), 4))
run("INT4, groups of 32, everything", lambda k: (lambda w: quant(w, 4, 32), 4.5))
run("INT4 blocks g32 zero-point + tables INT8", lambda k: (lambda w: quant_asym(w, 4, 32), 5) if k in BLOCK else (lambda w: quant_col(w, 8), 8))
run("INT4 blocks g32 symmetric + tables INT8", lambda k: (lambda w: quant(w, 4, 32), 4.5) if k in BLOCK else (lambda w: quant_col(w, 8), 8))
run("INT3 blocks g32 zero-point + tables INT8", lambda k: (lambda w: quant_asym(w, 3, 32), 4) if k in BLOCK else (lambda w: quant_col(w, 8), 8))
run("INT2 blocks g32 zero-point + tables INT8", lambda k: (lambda w: quant_asym(w, 2, 32), 3) if k in BLOCK else (lambda w: quant_col(w, 8), 8))
