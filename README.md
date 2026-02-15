# USENIX2026

Scripts used in the paper **“Detecting OODified Jailbreak Attacks via Hidden-Layer Representation Dynamics”**.

## Files
- `generate_benign.py`  
  Build **benign prompts** by rewriting harmful prompts with **intent neutralization** (async + batched, appends to CSV per batch).

- `generate hidden states.py`  
  Run the target LLM, generate responses, and **dump hidden states** for early tokens (and optionally tail tokens) for later analysis/training.

- `MiniLM_cluster.py`  
  Compute MiniLM embeddings and run clustering (e.g., HDBSCAN) to obtain **OOD-intent clusters**.

- `eval.py`  
  Evaluate a detector: pick thresholds by **target FPR** on benign data, then test detection on OODified attack sets.

