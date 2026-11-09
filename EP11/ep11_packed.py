"""EP11 · Real packed files. Saves the EP10 support bot as actual 8-bit and 4-bit files on disk,
measures each file's size, then loads the bot back FROM each file and re-asks the 14 chats it never trained on.
Packing: 8-bit = int8 weights + one float32 scale per column. 4-bit = two weights per byte (uint8),
groups of 32, each group with a float32 scale and an int8 zero point. 2-bit = four weights per byte, same groups. Biases and layer norms stay float32."""
import os, sys, torch, tiktoken
sys.path.insert(0, "."); import gpt2_torch
enc = tiktoken.get_encoding("gpt2"); EOT = enc.eot_token
m = gpt2_torch.load(); m.eval(); SD = torch.load("agent_ep10.pt"); m.load_state_dict(SD)
ORIG = {k: v.detach().clone() for k, v in m.p.items()}; MATS = [k for k, v in ORIG.items() if v.dim() == 2]
BLOCK = [k for k in MATS if "_h_" in k]
HO = [l.strip() for f in ("support_validation.txt", "support_test_final.txt") for l in open(f) if l.strip()]
os.makedirs("packed", exist_ok=True)

def pack8(w):   # int8 + float32 scale per column
    s = w.abs().amax(0, keepdim=True) / 127; return {"q": torch.round(w / s).clamp(-127, 127).to(torch.int8), "s": s.float()}
def unpack8(d): return d["q"].float() * d["s"]
def pack4(w):   # groups of 32, asymmetric: int8 zero point + float32 scale, two 4-bit values per byte
    g = w.reshape(-1, 32); lo = g.amin(1, keepdim=True); hi = g.amax(1, keepdim=True)
    s = (hi - lo) / 15; z = torch.round(-lo / s); q = (torch.round(g / s) + z).clamp(0, 15).to(torch.uint8).flatten()
    return {"q": (q[0::2] | (q[1::2] << 4)).contiguous(), "s": s.float(), "z": z.to(torch.int8), "shape": list(w.shape)}
def pack2(w):   # groups of 32, asymmetric: int8 zero point + float32 scale, four 2-bit values per byte
    g = w.reshape(-1, 32); lo = g.amin(1, keepdim=True); hi = g.amax(1, keepdim=True)
    s = (hi - lo) / 3; z = torch.round(-lo / s); q = (torch.round(g / s) + z).clamp(0, 3).to(torch.uint8).flatten()
    return {"q": (q[0::4] | (q[1::4] << 2) | (q[2::4] << 4) | (q[3::4] << 6)).contiguous(), "s": s.float(), "z": z.to(torch.int8), "shape": list(w.shape)}
def unpack2(d):
    b = d["q"]; q = torch.stack([b & 3, (b >> 2) & 3, (b >> 4) & 3, b >> 6], 1).flatten().float().reshape(-1, 32)
    return ((q - d["z"].float()) * d["s"]).reshape(d["shape"])
def unpack4(d):
    b = d["q"]; q = torch.stack([b & 15, b >> 4], 1).flatten().float().reshape(-1, 32)
    return ((q - d["z"].float()) * d["s"]).reshape(d["shape"])

PLANS = {
 "fp32":  lambda k, v: ("f32", v.clone()),
 "fp16":  lambda k, v: ("f16", v.half()) if k in MATS else ("f32", v.clone()),
 "int8":  lambda k, v: ("i8", pack8(v)) if k in MATS else ("f32", v.clone()),
 "mix4":  lambda k, v: (("i4", pack4(v)) if k in BLOCK else ("i8", pack8(v))) if k in MATS else ("f32", v.clone()),
 "mix2":  lambda k, v: (("i2", pack2(v)) if k in BLOCK else ("i8", pack8(v))) if k in MATS else ("f32", v.clone()),
}
UNPACK = {"f32": lambda d: d, "f16": lambda d: d.float(), "i8": unpack8, "i4": unpack4, "i2": unpack2}

@torch.no_grad()
def reply(q, n=30):
    ids = enc.encode(f"{q} Agent:"); N = len(ids)
    for _ in range(n):
        x = int(m(torch.tensor(ids))[-1].argmax())
        if x == EOT: break
        ids.append(x)
    return enc.decode(ids[N:]).strip()

BASE = None
for name, plan in PLANS.items():
    path = f"packed/support_bot_{name}.pt"
    torch.save({k: plan(k, v) for k, v in ORIG.items()}, path)
    size = os.path.getsize(path)
    loaded = torch.load(path)                                   # load the bot back from the file on disk
    with torch.no_grad():
        for k, (kind, d) in loaded.items(): m.p[k].copy_(UNPACK[kind](d))
    reps = [reply(c.split(" Agent:")[0]) for c in HO]
    if BASE is None: BASE = reps
    same = sum(a == b for a, b in zip(BASE, reps))
    print(f"{name:5s} | file on disk {size/1e6:6.1f} MB | {497.76 / (size/1e6):4.2f}x smaller than 497.8 MB | same reply as the original bot {same:2d}/14", flush=True)
    print(f"      'Where is my refund?' -> {reps[3]!r}", flush=True)
