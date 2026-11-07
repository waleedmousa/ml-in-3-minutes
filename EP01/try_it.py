# ML in 3 Minutes · 01 · Tokens & embeddings
# pip install tiktoken gensim
import tiktoken
import gensim.downloader as api

enc = tiktoken.get_encoding("gpt2")              # the real GPT-2 tokenizer
ids = enc.encode("I want a refund for a double charge")
print(ids)                                       # token IDs
print([enc.decode([i]) for i in ids])            # the pieces
print(enc.encode("refund"), enc.encode(" refund"))  # the space matters

glove = api.load("glove-wiki-gigaword-50")       # 400,000 words x 50 numbers
print(glove["refund"][:3])                       # the first 3 of 50
print(glove.most_similar("refund", topn=3))
print(glove.similarity("refund", "reimbursement"))
print(glove.similarity("refund", "banana"))
print(glove.most_similar("charge", topn=8))      # hello, part two
