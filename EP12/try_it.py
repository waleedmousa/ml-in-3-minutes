# EP12 · Why chatbots batch: one pass over the weights serves many tokens
import torch, time
torch.set_num_threads(2)
W = torch.randn(4096, 4096)              # one layer: 16.8 million weights = 67 MB
def ms(f, n=30):
    f(); t = time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter() - t) / n * 1000
for tokens in (1, 8, 64):                # 1 = one chat decoding; 64 = a batch (or a prompt)
    x = torch.randn(tokens, 4096)
    t = ms(lambda: x @ W)
    print(f"{tokens:3d} tokens at once: {t:6.2f} ms → {tokens / t * 1000:9,.0f} tokens/s")
