"""EP10 · Fine-tuning the real GPT-2 (124M) on 40 support chats we wrote.
Validation: 8 held-out chats, used to pick the stopping pass. Test: 6 more chats, scored once, only on the chosen checkpoint."""
import sys, time, random, torch, tiktoken
sys.path.insert(0, "."); import gpt2_torch
torch.manual_seed(0); random.seed(0)
enc = tiktoken.get_encoding("gpt2"); m = gpt2_torch.load()
TRAIN = [l.strip() for l in open("support_chats.txt") if l.strip()]
VAL = [l.strip() for l in open("support_validation.txt") if l.strip()]
TEST = [l.strip() for l in open("support_test_final.txt") if l.strip()]
GEN = ["The weather today is sunny and warm.", "She opened the door and walked into the kitchen.", "Paris is the capital of France."]
F = torch.nn.functional
EOT = enc.eot_token  # 50256, the end-of-text token
def loss_of(t, eot=True):
    ids = torch.tensor(enc.encode(t) + ([EOT] if eot else [])); return F.cross_entropy(m(ids)[:-1], ids[1:])
def avg(ts, eot=True):
    with torch.no_grad(): return sum(loss_of(t, eot).item() for t in ts) / len(ts)
def reply(q, n=28):
    ids = enc.encode(f"Customer: {q} Agent:")
    with torch.no_grad():
        for _ in range(n):
            nxt = m(torch.tensor(ids))[-1].argmax().item()
            if nxt == EOT: return enc.decode(ids).split("Agent:", 1)[1].strip() + "  [end]"
            ids.append(nxt)
    return enc.decode(ids).split("Agent:", 1)[1].strip() + " ..."

QS = ["I was charged twice for my train ticket.", "My card was swallowed by the machine.", "Where is my refund?", "There is a payment on my account I never made."]
log = []
def report(ep):
    r = dict(epoch=ep, train=round(avg(TRAIN), 3), validation=round(avg(VAL), 3), general=round(avg(GEN, False), 3)); log.append(r); print(r, flush=True)
report(0)
print("BEFORE:"); [print(" ", q, "->", reply(q)) for q in QS]
opt = torch.optim.AdamW(m.parameters(), lr=float(sys.argv[1]) if len(sys.argv) > 1 else 2e-5, weight_decay=0.0)
best = None; t0 = time.time()
for ep in range(1, int(sys.argv[2]) + 1 if len(sys.argv) > 2 else 7):
    order = TRAIN[:]; random.shuffle(order)
    for t in order:
        opt.zero_grad(); loss_of(t).backward(); opt.step()
    report(ep)
    if best is None or log[-1]["validation"] < best[0]:
        best = (log[-1]["validation"], ep); snap = {k: v.detach().clone() for k, v in m.state_dict().items()}
        print("AFTER epoch", ep, ":"); [print(" ", q, "->", reply(q)) for q in QS]
print("best validation epoch", best, "minutes", round((time.time() - t0) / 60, 1))
m.load_state_dict(snap)   # the checkpoint chosen by validation loss
print("FINAL TEST (scored once, chosen checkpoint, epoch", best[1], "): test loss", round(avg(TEST), 3), "| general-text loss", round(avg(GEN, False), 3))
with torch.no_grad():
    base = gpt2_torch.load()
    T0 = sum(F.cross_entropy(base(torch.tensor(enc.encode(t) + [EOT]))[:-1], torch.tensor(enc.encode(t) + [EOT])[1:]).item() for t in TEST) / len(TEST)
print("test loss of plain GPT-2 before fine-tuning:", round(T0, 3))
