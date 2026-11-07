# ML in 3 Minutes 02: attention, the bare-bones version
# pip install gensim numpy
import numpy as np, gensim.downloader as api
glove = api.load("glove-wiki-gigaword-50")
words = "i want a refund for a double charge".split()
X = np.array([glove[w] for w in words])
q = glove["charge"]
scores = X @ q / np.sqrt(50)              # 1. score every word
w = np.exp(scores) / np.exp(scores).sum() # 2. softmax: percentages
new = w @ X                               # 3. mix
for word, p in zip(words, w): print(f"{word:7s} {p:.1%}")
cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
for t in ["payment", "criminal"]:
    print(t, round(cos(q, glove[t]), 2), "to", round(cos(new, glove[t]), 2))
