# ML in 3 Minutes 04: count every number in GPT-2 small
# sizes from github.com/openai/gpt-2 src/model.py
vocab, ctx, d, heads, layers = 50257, 1024, 768, 12, 12
attention = (d * 3 * d + 3 * d) + (d * d + d)      # query, key, value + output
feed_forward = (d * 4 * d + 4 * d) + (4 * d * d + d)
norms = 2 * 2 * d
block = attention + feed_forward + norms
total = vocab * d + ctx * d + layers * block + 2 * d
print("head size:", d // heads)
print("one block:", f"{block:,}")
print("feed-forward share:", f"{feed_forward / block:.1%}")
print("token table share:", f"{vocab * d / total:.1%}")
print("total:", f"{total:,}")
