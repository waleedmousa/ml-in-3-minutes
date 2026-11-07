# ML in 3 Minutes 10: fine-tune the real GPT-2 (pip install torch tiktoken onnx; needs gpt2_torch.py + the GPT-2 ONNX file)
import random, torch, tiktoken, gpt2_torch
random.seed(0); enc, m = tiktoken.get_encoding("gpt2"), gpt2_torch.load()
chats = [l.strip() for l in open("support_chats.txt") if l.strip()]
def reply(q):
    ids = enc.encode(f"Customer: {q} Agent:")
    for _ in range(40):                     # write until the end marker
        with torch.no_grad(): nxt = m(torch.tensor(ids))[-1].argmax().item()
        if nxt == enc.eot_token: break
        ids.append(nxt)
    return enc.decode(ids).split("Agent:")[1]
print("before:", reply("My card was swallowed by the machine."))
opt = torch.optim.AdamW(m.parameters(), lr=2e-5, weight_decay=0.0)
for epoch in range(5):                      # 5 passes over 40 chats
    order = chats[:]; random.shuffle(order)
    for c in order:
        ids = torch.tensor(enc.encode(c) + [enc.eot_token])
        loss = torch.nn.functional.cross_entropy(m(ids)[:-1], ids[1:])
        opt.zero_grad(); loss.backward(); opt.step()
print("after: ", reply("My card was swallowed by the machine."))
