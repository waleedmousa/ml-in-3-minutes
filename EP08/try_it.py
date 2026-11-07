# ML in 3 Minutes 08: backpropagation is the chain rule, and you can check it
w, x, y = 1.5, 4.0, 8.0                 # weight, input, right answer
def loss(w): return (w * x - y) ** 2     # predict, then measure the error
pred = w * x                             # forward pass
dloss_dpred = 2 * (pred - y)             # how the loss reacts to the prediction
dpred_dw = x                             # how the prediction reacts to the weight
backprop = dloss_dpred * dpred_dw        # chain rule: multiply the two
nudge = (loss(w + 1e-4) - loss(w - 1e-4)) / 2e-4   # slow check: wiggle the weight
print("backprop slope:", backprop)
print("nudge check:   ", round(nudge, 4))
