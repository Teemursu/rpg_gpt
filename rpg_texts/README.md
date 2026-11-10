# rpg_texts: dialogue extraction and DialoGPT (2021-2022)

History from **`Teemursu/ma_thesis`** (private, Dec 2021 to Mar 2022), plus
working-copy changes from Jan-Feb 2023 that were never pushed. Those were
recovered from `R:\Documents\University\MA thesis\Code\rpg_texts\data`, the local
clone of that repository.

The game exports, the extracted CSV datasets and the persuasion-classifier
scripts (`models/classifier_model.py`, `models/multilingual_persuasion_detection.py`)
were filtered out of the history. The classifier belongs to the DiGRA 2022
*Multilingual Persuasion Detection* paper, which has its own repository. The
extraction tools here are shared with that paper: thesis §3.1 reuses its data
and method.

```
extracting_data/tools/   game-file -> dialogue-table -> dialogue-path scripts (see ../data/README.md)
models/train_dialogpt.py DialoGPT-large fine-tuning on the 10-turn context table (Rick&Morty-tutorial style script)
models/train_dialogpt_maybeold.py, talk_with_rpg_gpt.py, gpt_rpg.py (Colab export), dialogue_nlg.py (stub)
stats/                   number of rows that have context/N, per N
testdata.py              GPT-J data-loading test (Jan 2023)
```

## Run status

| script | status |
|---|---|
| `dialogpt_dataset_recursive.py`, `concat_dialog_files.py` | **runs** (smoke test on toy data). Paths made relative, run from the data root that contains `data/` |
| `models/train_dialogpt.py` | **runs** with `DIALOGPT_MODEL=sshleifer/tiny-gpt2 DIALOGPT_MAX_STEPS=2` (smoke test). Replaced the removed `AutoModelWithLMHead`/`transformers.AdamW`, added a CPU fallback |
| `kotor_xml.py`, `nwn_xml.py`, `sid_to_str.py`, `get_*.py`, `combine_*.py`, `data_processing.py` | compile only. They need the exported game files and Windows-style relative paths (`data\...`), so they were not run |
| `models/talk_with_rpg_gpt.py` | needs a local DialoGPT checkpoint (hard-coded `D:/Code/...` path) |

Known code issue: in `combine_datasets.py` the lines
`df.loc[df.persuasion == "Intellect"] = "Intelligence"` (and the `Spellcraft`
line) overwrite **every column** of the matching rows, not only `persuasion`.
That turns those dialogue lines into the bare word "Intelligence"/"Magic". The
thesis (§3.2) describes the intended renaming only.
