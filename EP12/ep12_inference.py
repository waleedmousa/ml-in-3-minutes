"""EP12 · LLM inference on the real GPT-2 (124M), 2 CPU cores, using the support agent fine-tuned in EP10 (agent_ep10.pt, made by make_agent.py).
Same weights as gpt2_torch.py, written as nn.Linear layers so we can (1) keep a KV cache,
(2) run many chats at once (batching) and (3) swap in 8-bit weights. Times are medians of repeated runs."""
import sys, time, math, statistics, torch, tiktoken, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "."); import gpt2_torch
torch.manual_seed(0); torch.set_num_threads(2)
enc = tiktoken.get_encoding("gpt2"); W = gpt2_torch.load_weights()
SD = torch.load("agent_ep10.pt"); W = {k: SD["p." + k.replace(".", "_")] if ("p." + k.replace(".", "_")) in SD else v for k, v in W.items()}
EOT = enc.eot_token
def lin(w, b): l = nn.Linear(w.shape[0], w.shape[1]); l.weight.data = w.float().T.contiguous(); l.bias.data = b.float(); return l
class Block(nn.Module):
    def __init__(s, i):
        super().__init__(); P = lambda k: W[f"transformer.h.{i}.{k}"]
        s.ln1 = (P("ln_1.weight"), P("ln_1.bias")); s.ln2 = (P("ln_2.weight"), P("ln_2.bias"))
        s.qkv = lin(P("attn.c_attn.weight"), P("attn.c_attn.bias")); s.proj = lin(P("attn.c_proj.weight"), P("attn.c_proj.bias"))
        s.fc = lin(P("mlp.c_fc.weight"), P("mlp.c_fc.bias")); s.out = lin(P("mlp.c_proj.weight"), P("mlp.c_proj.bias"))
    def forward(s, x, cache):                      # x: [B, T, 768]; cache: [k, v] or None
        B, T, _ = x.shape
        q, k, v = s.qkv(F.layer_norm(x, (768,), *s.ln1)).split(768, -1)
        q, k, v = [t.view(B, T, 12, 64).transpose(1, 2) for t in (q, k, v)]
        if cache is not None:
            if cache[0] is not None: k = torch.cat([cache[0], k], 2); v = torch.cat([cache[1], v], 2)
            cache[0], cache[1] = k, v
        a = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        x = x + s.proj(a.transpose(1, 2).reshape(B, T, 768))
        return x + s.out(F.gelu(s.fc(F.layer_norm(x, (768,), *s.ln2)), approximate="tanh"))
class GPT(nn.Module):
    def __init__(s):
        super().__init__(); s.wte = W["transformer.wte.weight"].float(); s.wpe = W["transformer.wpe.weight"].float()
        s.blocks = nn.ModuleList(Block(i) for i in range(12)); s.lnf = (W["transformer.ln_f.weight"], W["transformer.ln_f.bias"])
        s.head = nn.Linear(768, 50257, bias=False); s.head.weight.data = s.wte
    def forward(s, ids, caches=None, pos0=0):       # ids: [B, T]
        x = s.wte[ids] + s.wpe[pos0:pos0 + ids.shape[1]]
        for i, b in enumerate(s.blocks): x = b(x, caches[i] if caches else None)
        return s.head(F.layer_norm(x[:, -1:], (768,), *s.lnf))[:, -1]
m = GPT().eval()
import torch.ao.quantization as Q
HIST = [l.strip() for l in open("support_chats.txt") if l.strip()]
Q1 = "Customer: Where is my refund? Agent:"          # a validation question: never trained on
def med(f, r=5):
    ts = []
    for _ in range(r): t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return statistics.median(ts)
def new_cache(): return [[None, None] for _ in range(12)]
@torch.no_grad()
def gen(model, prompt_ids, G=40, cache=True):
    ids = list(prompt_ids); N = len(ids)
    if cache:
        cc = new_cache(); ids.append(int(model(torch.tensor([ids]), cc).argmax()))
        while ids[-1] != EOT and len(ids) < N + G: ids.append(int(model(torch.tensor([[ids[-1]]]), cc, len(ids) - 1).argmax()))
    else:
        ids.append(int(model(torch.tensor([ids])).argmax()))
        while ids[-1] != EOT and len(ids) < N + G: ids.append(int(model(torch.tensor([ids])).argmax()))
    return ids[N:-1] if ids[-1] == EOT else ids[N:]

with torch.no_grad():
    ref = gpt2_torch.load().eval(); ref.load_state_dict(SD)
    p = enc.encode(Q1); print(f"matches the EP10 agent: {int(ref(torch.tensor(p))[-1].argmax()) == int(m(torch.tensor([p])).argmax())}"); del ref

    # 1. the reply, streamed
    r = gen(m, p); print(f"PROMPT: {Q1!r} ({len(p)} tokens)")
    print(f"REPLY: {enc.decode(r)!r} ({len(r)} tokens, then the end-of-chat token)")
    print("ONE TOKEN AT A TIME:", " | ".join(enc.decode([t]) for t in r))

    # 2. prefill vs decode, for prompts of different length (history = earlier chats)
    print("--- prefill vs decode ---")
    for n_hist in (0, 4, 12):
        ids0 = enc.encode(" ".join(HIST[:n_hist] + [Q1]) if n_hist else Q1); N = len(ids0)
        def prefill(): c = new_cache(); m(torch.tensor([ids0]), c); return c
        t_pre = med(prefill); c = prefill(); nx = torch.tensor([[int(m(torch.tensor([ids0])).argmax())]])
        def step(): m(nx, [[k, v] for k, v in c], N)
        t_dec = med(step, 15)
        print(f"prompt {N:4d} tokens: PREFILL (all at once) {1000*t_pre:6.0f} ms = {1000*t_pre/N:4.1f} ms per token | DECODE one new token {1000*t_dec:5.1f} ms -> {1/t_dec:3.0f} tokens/s")
    LONG = ids0; NL = len(LONG)

    # 3. KV cache: the same reply, after a long conversation, with and without the cache
    lp = LONG; a = gen(m, lp, cache=False); b = gen(m, lp, cache=True)
    t_nc = med(lambda: gen(m, lp, cache=False), 3); t_c = med(lambda: gen(m, lp, cache=True), 3)
    print(f"--- KV cache ({NL}-token chat, {len(b)}-token reply, identical: {a == b}) ---")
    print(f"WITHOUT cache (re-read the whole chat for every token): {t_nc:.2f} s | WITH KV cache: {t_c:.2f} s | {t_nc/t_c:.1f}x faster")
    per_tok = 2 * 12 * 768 * 4
    print(f"KV cache per token (FP32): 2 x 12 layers x 768 x 4 bytes = {per_tok:,} bytes | {NL}-token chat: {per_tok*NL/1e6:.1f} MB | 1,024 tokens: {per_tok*1024/1e6:.1f} MB")

    # 4. every new token reads every weight
    wbytes = sum(t.numel() * 4 for t in m.parameters()); c = new_cache(); m(torch.tensor([p]), c); nx = torch.tensor([[r[0]]])
    t1 = med(lambda: m(nx, [[k, v] for k, v in c], len(p)), 15)
    print(f"--- memory ---\nweights read per new token (FP32): {wbytes/1e6:.0f} MB | one token {1000*t1:.1f} ms -> {1/t1:.0f} tokens/s = {wbytes/t1/1e9:.1f} GB/s of weights")

    # 5. batching: B customers, one decode step each
    print("--- batching ---")
    for B in (1, 8, 32, 64):
        cB = [[k.expand(B, -1, -1, -1).contiguous(), v.expand(B, -1, -1, -1).contiguous()] for k, v in c]; x = nx.expand(B, 1)
        t = med(lambda: m(x, [[k, v] for k, v in cB], len(p)), 7)
        print(f"BATCH {B:2d} chats: one step {1000*t:6.1f} ms -> {B/t:5.0f} tokens/s total, {1/t:5.1f} tokens/s per chat")

    # 6. 8-bit weights (the 12 blocks; token table stays FP32, see EP11)
    q = Q.quantize_dynamic(m, {"blocks": Q.per_channel_dynamic_qconfig}, dtype=torch.qint8)
    n8 = sum(t.numel() for n, t in m.named_parameters() if n.startswith("blocks") and n.endswith("weight"))
    qbytes = wbytes - n8 * 3
    cq = new_cache(); q(torch.tensor([p]), cq); tq = med(lambda: q(nx, [[k, v] for k, v in cq], len(p)), 15)
    print(f"--- INT8 ---\nweights read per token: {qbytes/1e6:.0f} MB (was {wbytes/1e6:.0f}) | one token {1000*tq:.1f} ms -> {1/tq:.0f} tokens/s ({t1/tq:.1f}x faster)")
    print(f"INT8 REPLY: {enc.decode(gen(q, p))!r}")
    HO = [l.strip() for f in ("support_validation.txt", "support_test_final.txt") for l in open(f) if l.strip()]
    same = sum(gen(m, enc.encode(l.split(" Agent:")[0] + " Agent:")) == gen(q, enc.encode(l.split(" Agent:")[0] + " Agent:")) for l in HO)
    print(f"INT8 gives the identical reply on {same} of {len(HO)} chats it never trained on")
