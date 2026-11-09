"""GPT-2 small (124M) in plain PyTorch, weights loaded from the ONNX Model Zoo file gpt2-lm-head-10.onnx.
Same architecture as openai/gpt-2 model.py: 12 blocks, 12 heads, 768 dims, tied token table."""
import os, math, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
ONNX = os.environ.get("GPT2_ONNX", os.path.join(HERE, "gpt2-lm-head-10.onnx"))
PT = os.path.join(os.path.dirname(ONNX), "gpt2_from_onnx.pt")

def load_weights():
    if os.path.exists(PT): return torch.load(PT)
    import onnx
    from onnx import numpy_helper
    m = onnx.load(ONNX)
    W = {t.name: torch.tensor(numpy_helper.to_array(t).copy()) for t in m.graph.initializer if t.name.startswith("transformer.")}
    torch.save(W, PT); return W

class GPT2(torch.nn.Module):
    def __init__(self, W):
        super().__init__()
        self.p = torch.nn.ParameterDict({k.replace(".", "_"): torch.nn.Parameter(v.float()) for k, v in W.items() if not k.endswith(".attn.bias") and not k.endswith("masked_bias")})
    def g(self, k): return self.p[k.replace(".", "_")]
    def forward(self, ids):                       # ids: [T] -> logits [T, 50257]
        T = ids.shape[0]; P = self.g
        x = P("transformer.wte.weight")[ids] + P("transformer.wpe.weight")[:T]
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        for i in range(12):
            b = f"transformer.h.{i}."
            h = F.layer_norm(x, (768,), P(b + "ln_1.weight"), P(b + "ln_1.bias"), 1e-5)
            q, k, v = (h @ P(b + "attn.c_attn.weight") + P(b + "attn.c_attn.bias")).split(768, dim=-1)
            q, k, v = [t.view(T, 12, 64).transpose(0, 1) for t in (q, k, v)]
            a = (q @ k.transpose(-1, -2)) / math.sqrt(64)
            a = a.masked_fill(~mask, float("-inf")).softmax(-1)
            o = (a @ v).transpose(0, 1).reshape(T, 768)
            x = x + o @ P(b + "attn.c_proj.weight") + P(b + "attn.c_proj.bias")
            h = F.layer_norm(x, (768,), P(b + "ln_2.weight"), P(b + "ln_2.bias"), 1e-5)
            h = F.gelu(h @ P(b + "mlp.c_fc.weight") + P(b + "mlp.c_fc.bias"), approximate="tanh")
            x = x + h @ P(b + "mlp.c_proj.weight") + P(b + "mlp.c_proj.bias")
        x = F.layer_norm(x, (768,), P("transformer.ln_f.weight"), P("transformer.ln_f.bias"), 1e-5)
        return x @ P("transformer.wte.weight").T

def load():
    torch.manual_seed(0); return GPT2(load_weights())
