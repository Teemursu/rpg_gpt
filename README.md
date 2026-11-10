# RPG-GPT: Leveling up game dialogue with creative NLG

Code for the Master's thesis of **Teemu Pöyhönen** (2023), Master's Programme in
Linguistic Diversity and Digital Humanities (Language Technology), Faculty of Arts,
University of Helsinki. Supervisors: **Mika Hämäläinen** and **Khalid Alnajjar**.

- Published version (Helda): https://helda.helsinki.fi/items/5e2f7523-c61f-4315-814d-47b8a0e23d06
- PDF: [`docs/Poyhonen_2023_RPG-GPT_MA_thesis.pdf`](docs/Poyhonen_2023_RPG-GPT_MA_thesis.pdf) (24 Apr 2023, 70 pp.)

## Summary

The thesis asks whether natural language generation can write dialogue for
non-player characters in role-playing games. Dialogue was extracted from six
RPGs: Knights of the Old Republic 1 & 2, Neverwinter Nights, Pillars of Eternity
1 & 2 and Tyranny. The reply trees were unrolled recursively into dialogue paths
and turned into **847,038 context/response pairs** (about 128 M words), with
NPC turns as targets. These pairs were used to fine-tune Microsoft's
**GODEL-v1.1-large** (T5-based, 770M parameters), giving "RPG-GPT". Earlier
attempts used DialoGPT, GPT-Neo and GPT-J.

The evaluation follows computational-creativity frameworks (creative tripod:
skill, appreciation, imagination) and has two parts:
- a Qualtrics survey with 86 complete responses comparing 7 game dialogues with
  7 RPG-GPT dialogues on relevance/intention (skill) and focused attention
  (appreciation): Wilcoxon tests, correlations, Bartlett/KMO and exploratory factor analysis;
- an interactive study with 6 participants.

RPG-GPT's answers were rated almost as relevant as the game's own
(3.85 vs 3.93, p = 0.036), and participants reported engagement.

## Repository structure

```
rpg_texts/      2021-22  game-dialogue extraction tools + DialoGPT prototype    (history of Teemursu/ma_thesis)
rpg_bot/        Jan 2023 DialoGPT/GPT-Neo/GPT-J scaling attempts on CSC Mahti   (recovered, never versioned)
rpg_bot_mahti/  2023     final GODEL fine-tuning, chat script, survey analysis  (history of Teemursu/rpg_gpt + recovered final code)
smoke_test/     end-to-end smoke test on invented toy dialogue
data/README.md  how NLG_RPG was built, where the data, checkpoints and logs are kept
docs/           thesis PDF
```

Each part has its own README with run status and notes on how the code compares
with the thesis. Pipeline order: `rpg_texts/extracting_data/tools` (steps 1-5 in
`data/README.md`), then `rpg_bot_mahti/data` (steps 6-7), then
`rpg_bot_mahti/godel_finetuning.py`, then `talk_w_godel.py`, and finally
`rpg_bot_mahti/evaluation`.

## Data

The game dialogue is copyrighted. As stated in thesis §3.3, the dataset and the
fine-tuned models are **not released**. They stay on the author's archive drive
(paths are in [`data/README.md`](data/README.md)). The only data in git is the
anonymised survey responses: Likert items only, with no IPs, locations,
timestamps or free text.

## Running

```bash
python -m venv .venv && . .venv/Scripts/activate      # Python 3.12
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python smoke_test/run_smoke_test.py                    # ~1 min on CPU, downloads two tiny HF models
```

The smoke test builds dialogue trees from a toy dialogue table. It then makes the
prompt/completion jsonl, runs a 2-step GODEL fine-tune and a chat with
`hf-internal-testing/tiny-random-T5ForConditionalGeneration`, a 2-step DialoGPT
fine-tune with `sshleifer/tiny-gpt2`, and the survey analysis scripts. Full
training needs `data/NLG_RPG.jsonl` and a large GPU. The original runs were on
CSC's Mahti cluster (A100 GPUs), and the Slurm scripts in `rpg_bot*/` show the
setup.

Minimal compatibility fixes were applied for Python 3.12, transformers 5 and
pandas 3. Each one is marked with a `2026` comment and listed in the git log.

## Citation

```bibtex
@mastersthesis{poyhonen2023rpggpt,
  author = {P{\"o}yh{\"o}nen, Teemu},
  title  = {{RPG-GPT}: Leveling up game dialogue with creative {NLG}},
  school = {University of Helsinki, Faculty of Arts},
  type   = {Master's thesis},
  year   = {2023},
  url    = {https://helda.helsinki.fi/items/5e2f7523-c61f-4315-814d-47b8a0e23d06}
}
```

Related: Pöyhönen, Hämäläinen & Alnajjar (2022), *Multilingual Persuasion
Detection: Video Games as an Invaluable Data Source for NLP* (DiGRA 2022). The
extraction method and the KotOR/NWN data come from that paper (separate repository).

## History

The two GitHub repositories were merged with full history using
`git filter-repo`. Game files, datasets, the persuasion-classifier scripts and
editor/venv junk were removed. Code that existed only on local disks was added in
separate "recovered from ..." commits.
