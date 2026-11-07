# ML in 3 Minutes 07: gradient descent learns a 2% transfer fee from 4 examples
import numpy as np
amount = np.array([100, 250, 400, 800]) / 100      # in hundreds
fee = np.array([2, 5, 8, 16])                      # what the bank charged
for lr in [0.01, 0.05]:                            # small steps vs big steps
    w = 0.0                                        # start: guess a 0% fee
    for step in range(8):
        loss = np.mean((w * amount - fee) ** 2)
        slope = np.mean(2 * amount * (w * amount - fee))
        print(f"lr {lr} step {step}: fee rate {w:.2f}%  loss {loss:.2f}")
        w -= lr * slope                            # walk downhill
