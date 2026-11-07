# ML in 3 Minutes 03: attention is blind to order, positions fix it
# pip install gensim numpy
import numpy as np, gensim.downloader as api
glove = api.load("glove-wiki-gigaword-50")
def pe(n, d=50):                       # sine waves from "Attention Is All You Need"
    pos, i = np.arange(n)[:, None], np.arange(0, d, 2)
    P = np.zeros((n, d)); P[:, 0::2] = np.sin(pos / 10000 ** (i / d)); P[:, 1::2] = np.cos(pos / 10000 ** (i / d)); return P
def attend(X):
    S = X @ X.T / np.sqrt(50); W = np.exp(S) / np.exp(S).sum(1, keepdims=True); return W @ X
cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
A = "send 100 from savings to checking".split(); B = "send 100 from checking to savings".split()
XA, XB = np.array([glove[w] for w in A]), np.array([glove[w] for w in B])
print("no positions:", round(cos(attend(XA)[3], attend(XB)[5]), 6))
print("with positions:", round(cos(attend(XA + pe(6))[3], attend(XB + pe(6))[5]), 3))
