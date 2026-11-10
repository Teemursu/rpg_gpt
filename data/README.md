# Data and models (not in git)

Nothing in this folder is versioned. The dialogue data is copyrighted game
text. The thesis (section 3.3) explicitly chose **not** to release the data or
the fine-tuned models, so they stay on the owner's archive drive (`R:`). No
public download exists. `smoke_test/run_smoke_test.py` runs the whole pipeline
on invented toy dialogue instead.

## How the NLG_RPG dataset was built

Sources are six RPGs: KotOR 1 & 2 and Neverwinter Nights (BioWare Aurora/Odyssey
engine, `.dlg`/`.tlk` converted to XML), and Pillars of Eternity 1 & 2 and
Tyranny (Obsidian, JSON/XML conversation and stringtable files). The thesis
calls these "Infinity Engine" and "Eternity Engine" games.

| step | script | output (on R:) |
|---|---|---|
| 1. export game files (tlk2xml / xoreos tools, KotOR Tool, gff/dlg editors) | manual, tools in `MA thesis\Code\rpg_texts\tools`, `WeiDU-Linux` | `rpg_texts\data\extracting_data\{infinity,eternity}_engine\` (1.3 GB) |
| 2. dialogue line + reply IDs + speaker + skill check, per game | `rpg_texts/extracting_data/tools/kotor_xml.py`, `nwn_xml.py`, `sid_to_str.py`, `get_poe2_ids.py`, `get_tyranny_poe1_ids.py`, `get_poe_text_from_ids.py`, `get_*_text_*.py`, `combine_speaker_id_to_replies.py` | `rpg_texts\data\data\*.csv`, `*_dataset_with_speakers.csv` |
| 3. clean text, unify skill checks, special tokens (`<\|playername\|>` ...) and merge games | `combine_datasets.py` (this is the same table as the Multilingual Persuasion dataset of the DiGRA 2022 paper, with speakers added) | `rpg_texts\data\persuasion_dataset_w_quotes_and_squarebrackets_and_more_skills_with_speakers.csv` (68 MB) |
| 4. recursive walk of every reply tree into dialogue paths | `dialogpt_dataset_recursive.py` | `rpg_texts\data\data\to_concat_with_speakers\` (2,605 files) |
| 5. concatenate | `concat_dialog_files.py` | `rpg_bot_mahti\data\NLG_clean_with_speakers.csv` (991 MB, columns `response, context/0..N, game`) |
| 6. prompt/completion pairs (every NPC turn becomes a completion, the preceding turns are the prompt, joined with ` [SEP] `), filtering of `nan` and all-caps console/system lines, de-duplication | `rpg_bot_mahti/data/convert_to_json.py` | `rpg_bot_mahti\data\NLG_RPG.jsonl` (771 MB, **847,038 pairs**, matching the thesis) |
| 7. (optional) re-filter | `rpg_bot_mahti/data/data_clean.py` | `NLG_RPG_cleaned.jsonl` (129 MB, 309,318 pairs) |

All `R:` paths are relative to `R:\Documents\University\MA thesis\Code\`.

### Other dataset files on R:

- `Code\NLG_RPG.jsonl`: 1.24 GB, 1,433,286 pairs, 6 Feb 2023. An earlier and larger
  build of step 6, made before the final de-duplication/filter.
- `rpg_bot_mahti\data\NLG_RPG.csv` (992 MB), `NLG_RPG_cleaned_prepared*.jsonl`
  (132 MB, OpenAI-style `\n\n###\n\n` separators for the GPT-J runs),
  `rpg_bot_mahti\backups\NLG_GODEL.csv` (753 MB, `Context/Knowledge/Response` columns
  for Microsoft's `train_godel.py`).
- `rpg_bot\data\nlg_data_10_turns_all_data.csv` (857 MB) and
  `..._without_consoledialogue.csv` (463 MB): the 10-turn context tables used by
  the DialoGPT / GPT-Neo / GPT-J scripts.
- `rpg_texts\data\NLG_shorter_conv_w_old_files.csv` (38 MB), `persuasion_dataset_*.csv`
  variants (63-80 MB), `poe1/poe2/tyranny_dataset_with_speakers.csv`.

## Models / checkpoints (not in git)

| model | path on R: | notes |
|---|---|---|
| RPG-GPT (GODEL-v1.1-large, 770M) epoch 1 | `Code\rpg_bot_mahti\results\checkpoint-6618` | 2.8 GB, T5ForConditionalGeneration |
| RPG-GPT epoch 2 | `Code\rpg_bot_mahti\results\checkpoint-13236` | 2.8 GB, used by `talk_w_godel.py` |
| DialoGPT-small fine-tune (Oct 2022) | `Code\rpg_texts\data\output-small\checkpoint-3500` | 510 MB, the "previous DialoGPT experiment" in thesis §4.1.1 |

The thesis says the model chosen for the evaluation was trained for **4 epochs**.
No epoch 3 or 4 checkpoint was found locally. It probably stayed on CSC Mahti
(`/scratch/project_2001403/poyhnent/`), which is no longer accessible.

Training logs: `rpg_bot_mahti\logs\` (TensorBoard), `rpg_bot_mahti\mlruns\`
(MLflow, 70 runs), `rpg_texts\data\runs\` (DialoGPT TensorBoard),
`rpg_bot_mahti\backups\gptrpg_gptj.out.2019906` (68 MB GPT-J Slurm log).

## Survey data

Raw Qualtrics exports are kept out of git because they contain IP addresses,
geolocation, timestamps and response IDs:
`R:\Documents\University\MA thesis\Dialogue Engagement & Relevance survey_*.csv`,
`survey_data.csv`, `survey_data86.xlsx`, and
`Code\rpg_bot_mahti\evaluation\survey_data*.csv`. Individual response printouts
`R_*.pdf` are also kept out.
`rpg_bot_mahti/evaluation/survey_data861_anonymised.csv` (in git) holds the 86
completed responses: the Likert items plus the "games played" answer, in shuffled order.
`anonymise_survey.py` regenerates it:

    python rpg_bot_mahti/evaluation/anonymise_survey.py "<raw survey_data861.csv>" rpg_bot_mahti/evaluation/survey_data861_anonymised.csv
