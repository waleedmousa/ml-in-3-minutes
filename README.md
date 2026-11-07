# ML in 3 Minutes · code

Runnable code for every episode of **ML in 3 Minutes** on Runtime Explained. Each folder has the snippet shown at the end of the episode (`try_it.py`) and the output we got when we ran it (`try_it_output.txt`).

- **EP01** · Transformers Explained: Tokens & Embeddings
- **EP02** · Self-Attention Explained: Query, Key & Value
- **EP03** · Positional Encoding Explained: Why Word Order Matters
- **EP04** · Transformer Blocks Explained: Multi-Head Attention
- **EP05** · Next-Token Prediction Explained: How LLMs Write
- **EP06** · Loss Functions Explained: How Wrong Is the Model?
- **EP07** · Gradient Descent Explained: How Neural Networks Learn
- **EP08** · Backpropagation Explained: The Chain Rule in Action
- **EP09** · Overfitting Explained: Train vs Validation vs Test
- **EP10** · Fine-Tuning Explained: Train GPT-2 on Your Own Data

## The GPT-2 model file
EP02–EP06 and EP10 use the real GPT-2 small model from the ONNX Model Zoo (about 665 MB): https://github.com/onnx/models/tree/main/validated/text/machine_comprehension/gpt-2 — download `gpt2-lm-head-10.onnx` and put it next to the script you run.

## EP10 · fine-tuning
`EP10/try_it.py` needs `gpt2_torch.py` and `support_chats.txt` in the same folder (both included), plus the ONNX file above. `pip install torch tiktoken onnx`. About 4 minutes on 2 CPU cores.
`ep10_finetune.py` is the full experiment from the video: 40 training chats, 8 validation chats (used to pick the stopping pass), 6 test chats (scored once at the end).

Big tech ideas. Broken down. Put to work.
