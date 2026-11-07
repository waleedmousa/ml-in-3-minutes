# ML in 3 Minutes 05: real GPT-2 picks the next token
# pip install onnxruntime tiktoken numpy
# model (665 MB): https://github.com/onnx/models  validated/text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.onnx
import numpy as np, onnxruntime as ort, tiktoken
enc = tiktoken.get_encoding("gpt2")
gpt2 = ort.InferenceSession("gpt2-lm-head-10.onnx")
text = "I was charged twice, so I want a"
for T in [1.0, 0.5]:
    ids = enc.encode(text)
    scores = gpt2.run(None, {"input1": np.array([[ids]])})[0][0, 0, -1] / T
    p = np.exp(scores - scores.max()); p /= p.sum()
    print(f"temperature {T}:", [(enc.decode([i]), f"{p[i]:.1%}") for i in np.argsort(-p)[:5]])
for _ in range(8):                          # write, one token at a time
    ids = enc.encode(text)
    scores = gpt2.run(None, {"input1": np.array([[ids]])})[0][0, 0, -1]
    text += enc.decode([int(scores.argmax())])
print(text)
