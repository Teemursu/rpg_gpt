# rpg_bot: DialoGPT / GPT-Neo / GPT-J on CSC Mahti (Jan 2023)

This is a snapshot of a workspace that was never under version control:
`R:\Documents\University\MA thesis\Code\rpg_bot`. It covers the step between
the DialoGPT prototype (`../rpg_texts/models`) and the final GODEL model
(`../rpg_bot_mahti`). It holds the same DialoGPT-style training loop pointed at
larger causal LMs (EleutherAI GPT-Neo 2.7B, GPT-J 6B), with single-GPU, multi-GPU
(`torch.distributed.run`) and DeepSpeed variants, plus Slurm job scripts for
Mahti/Puhti (`project_2001403`).

`code_root/` holds the copies of `train_dialogpt.py`, `run.sh` and
`talk_with_rpg_gpt.py` that sat directly in `MA thesis\Code\`.
`eval_results.txt` (perplexity 4.82) is the only result that survives. It is
dated 10 Oct 2022, the same day as the DialoGPT-small `checkpoint-3500`, so it
most likely belongs to that run.

The following were left out: an ssh public key, a vendored `mpi4py` source
tree, and the 10-turn CSV datasets (0.5-0.9 GB, see `../data/README.md`).

Run status: not run. The scripts need CUDA/DeepSpeed/a Slurm cluster and
multi-GB models, and the thesis did not use them in the end. Their training loop
is the same as `../rpg_texts/models/train_dialogpt.py`, which the smoke test
covers.
