# rpg_bot_mahti: RPG-GPT (GODEL fine-tuning) and evaluation (2023)

History from **`Teemursu/rpg_gpt`** (public, Jan-Feb 2023: GPT-J/GPT-Neo/Pythia
DeepSpeed experiments). On top of it sits the **final, never-pushed** working copy
from `R:\Documents\University\MA thesis\Code\rpg_bot_mahti` (Feb-Apr 2023). That
copy is the code behind the thesis.

```
data/convert_to_json.py    dialogue paths -> prompt/completion jsonl (thesis §3.2.2)
data/data_clean.py         drop pairs with nan / all-caps console lines
data/data_expl.py          word count of the prompts
convert_json_to_csv.py     jsonl -> Context/Knowledge/Response csv for train_godel.py
godel_finetuning.py        fine-tune microsoft/GODEL-v1_1-large-seq2seq with HF Trainer  (thesis §4.1.1)
talk_w_godel.py            interactive chat with a checkpoint (temperature 0.9, repetition penalty 1)
train_godel.py             Microsoft's GODEL reference training script (MIT), copied, not used for the final model
evaluation/                survey analysis (thesis §4.2.3, §5.1)
backups/                   GPT-J / GPT-Neo / Pythia scripts (incl. LUMI variants) and two small Slurm logs
configs/ds_config.json     DeepSpeed ZeRO-2 config
```

## Run

```bash
# fine-tune (thesis setting: 128 per device, wd 0.01, adamw_torch, grad. checkpointing, save every epoch)
python godel_finetuning.py                     # needs data/NLG_RPG.jsonl and a big GPU
# smoke-sized: tiny random T5, 2 steps
RPG_GPT_BASE_MODEL=hf-internal-testing/tiny-random-T5ForConditionalGeneration \
RPG_GPT_MAX_STEPS=2 RPG_GPT_BATCH_SIZE=2 python godel_finetuning.py
RPG_GPT_CHECKPOINT=results python talk_w_godel.py

# survey analysis (from rpg_bot_mahti/), uses evaluation/survey_data861_anonymised.csv
python evaluation/eval_exploration.py      # per-dialogue Wilcoxon -> dialog_stats.csv
python evaluation/eval_explorationall.py   # pooled game vs AI + factor analysis -> dialog_stats_alll.csv
python evaluation/pca.py                   # correlation matrix, Bartlett, KMO, 4-factor EFA
```

All of these run in `../smoke_test/run_smoke_test.py`. Running the scripts on
the anonymised responses reproduces the committed `dialog_stats_alll.csv` exactly,
and the W/p rows of `dialog_stats.csv`.

## Code vs. thesis

- Data: `NLG_RPG.jsonl` has **847,038** pairs, as reported. Players and NPCs are
  marked `Player:`/`NPC:` and turns are joined with `[SEP]`. Skill checks are
  special tokens.
- Fine-tuning: the hyper-parameters match §4.1.1, except
  `num_train_epochs=20` in code vs. 10 in the text. Left truncation of the
  prompt and right truncation of the completion at 256 tokens match.
  Only `<|playername|>` is added as a special token; `<|playerrace|>` and
  `<|playerculture|>` (§3.2) appear in the data but are not added to the vocabulary.
- Labels are padded with the pad token rather than `-100`, so padding is part of the loss.
  That probably explains the very low training loss (0.12 after 2 epochs).
- Model selection: the thesis says 4 epochs. Only the epoch 1 and 2 checkpoints
  survive locally, and `talk_w_godel.py` loads epoch 2 (`checkpoint-13236`).
- Inference prompt: `talk_w_godel.py` prepends a GODEL-style
  `Instruction: ... [CONTEXT] ... =>` prefix and ends with ` NPC: `. The training
  prompts had no instruction or `[CONTEXT]` tag, so train and test formats differ.
- Statistics: the thesis uses the Wilcoxon signed-rank test. The per-dialogue
  results (`dialog_stats.csv`) were made with `wilcoxon`, but the pooled
  "all dialogues" script in its last version uses `ttest_ind`. The abstract's
  *3.93 vs 3.85, p = 0.0364* is reproduced by a Wilcoxon test on the pooled
  **skill** items (relevance + intention: 3.932 vs 3.846, p = 0.036). The pooled
  t-test in `dialog_stats_alll.csv` gives p = 0.063 for the same comparison.
