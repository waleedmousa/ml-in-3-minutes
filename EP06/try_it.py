# ML in 3 Minutes 06: the loss = how surprised the model is by the right answer
# pip install onnxruntime tiktoken numpy
# model (665 MB): https://github.com/onnx/models  validated/text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.onnx
import numpy as np, onnxruntime as ort, tiktoken
enc = tiktoken.get_encoding("gpt2")
gpt2 = ort.InferenceSession("gpt2-lm-head-10.onnx")
ids = enc.encode("I was charged twice, so I want a")
scores = gpt2.run(None, {"input1": np.array([[ids]])})[0][0, 0, -1]
p = np.exp(scores - scores.max()); p /= p.sum()
for word in [" refund", " lawyer", " pizza"]:
    q = p[enc.encode(word)[0]]
    print(f"answer{word!r:10} chance {q:.4%}  loss {-np.log(q):.2f}")
print("knows nothing (random guess):", round(np.log(len(p)), 2))
