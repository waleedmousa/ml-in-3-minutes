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
- **EP11** · Quantization Explained: How to Shrink an LLM
- **EP12** · LLM Inference Explained: Why Chatbots Type One Word at a Time

## The GPT-2 model file
EP02–EP06 and EP10–EP12 use the real GPT-2 small model from the ONNX Model Zoo (about 665 MB): https://github.com/onnx/models/tree/main/validated/text/machine_comprehension/gpt-2 — download `gpt2-lm-head-10.onnx` and put it next to the script you run.

## EP10 · fine-tuning
`EP10/try_it.py` needs `gpt2_torch.py` and `support_chats.txt` in the same folder (both included), plus the ONNX file above. `pip install torch tiktoken onnx`. About 4 minutes on 2 CPU cores.
`ep10_finetune.py` is the full experiment from the video: 40 training chats, 8 validation chats (used to pick the stopping pass), 6 test chats (scored once at the end).

## EP11 · quantization and EP12 · inference
`try_it.py` in both folders needs only `pip install torch` and runs in seconds.
The full experiments use the support bot from EP10. Build it once with `make_agent.py` (writes `agent_ep10.pt`, about 4 minutes on 2 CPU cores; needs the ONNX file above, `gpt2_torch.py` and the three `support_*.txt` files). Then:
- `EP11/ep11_agent_quant.py` – 16, 8, 4, 3 and 2 bits, one scale vs groups of 32, tested on 14 chats the bot never trained on.
- `EP11/ep11_agent_stats.py` – the outlier numbers: biggest weight vs typical, and how many weights round to zero at 4 bits.
- `EP11/ep11_packed.py` – saves the bot as real packed 8-bit, 4-bit and 2-bit files, measures each file, loads the bot back from it and re-tests.
- `EP12/ep12_inference.py` – prefill vs decode, KV cache, batching and 8-bit weights. Timings change from run to run; `ep12_run1-3.txt` are our three runs, and the video uses the median.

Big tech ideas. Broken down. Put to work.
