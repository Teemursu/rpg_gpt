"""End-to-end smoke test of the RPG-GPT pipeline on synthetic toy data.

Runs every stage of the thesis pipeline that does not need the original game
files, each in a throw-away working directory laid out the way the original
scripts expect (they use relative paths):

  1. rpg_texts/extracting_data/tools/dialogpt_dataset_recursive.py
       dialogue-line table -> one CSV of dialogue paths per dialogue file
  2. rpg_texts/extracting_data/tools/concat_dialog_files.py
       -> NLG_clean_with_speakers.csv
  3. rpg_bot_mahti/data/convert_to_json.py
       -> NLG_RPG.jsonl (prompt/completion pairs, NPC turns as completions)
  4. rpg_bot_mahti/data/data_clean.py           -> NLG_RPG_cleaned.jsonl
  5. rpg_bot_mahti/godel_finetuning.py          (tiny random T5, 2 steps)
  6. rpg_bot_mahti/talk_w_godel.py              (piped player input)
  7. rpg_texts/models/train_dialogpt.py         (tiny GPT-2, 2 steps)
  8. rpg_bot_mahti/evaluation/eval_explorationall.py and pca.py
       (on the committed anonymised survey responses)

The toy dialogue lines are invented; no game text is used.

Usage (from the repository root, inside the venv):
    python smoke_test/run_smoke_test.py [--keep] [--only STAGE ...]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parents[1]
TINY_T5 = os.environ.get("SMOKE_T5", "hf-internal-testing/tiny-random-T5ForConditionalGeneration")
TINY_GPT2 = os.environ.get("SMOKE_GPT2", "sshleifer/tiny-gpt2")
PLAYER_UUID = "b1a8e901-0000-0000-0000-000000000000"  # PoE player speaker GUID
PERSUASION_CSV = "persuasion_dataset_w_quotes_and_squarebrackets_and_more_skills_with_speakers.csv"


def toy_dialogue_table():
    """Rows in the format of the speaker-annotated persuasion dataset."""
    rows = []
    kotor = [  # sid, speaker, text, replies, skill
        (1, "Npc", "Halt, stranger. Few travellers come this far into the ruins.", "[2, 3, 4]", "False"),
        (2, "PC", "I am looking for the old archive. Do you know the way?", "[5]", "False"),
        (3, "PC", "Step aside. You do not want to stand in my way.", "[6]", "Persuade"),
        (4, "PC", "Never mind, I will find it myself.", "[]", "False"),
        (5, "Npc", "The archive lies beyond the collapsed bridge. Mind the guards.", "[7]", "False"),
        (6, "Npc", "All right, all right. No need for threats. Go on through.", "[]", "False"),
        (7, "PC", "Thank you. What guards?", "[8]", "False"),
        (8, "Npc", "Constructs, mostly. Old, but they still hit hard.", "[]", "False"),
    ]
    for sid, spk, text, replies, skill in kotor:
        rows.append(dict(game="KotOR 1", filename="toy/toy_guard.dlg.xml", id=sid, sid=float(sid),
                         replies=replies, speaker=spk, persuasion=skill, en=text))
    poe2 = [  # id, speaker, text, replies, skill
        ("0", "npc-merchant", "Welcome to my stall, traveller. Spices from across the southern sea!", "['1', '2']", "False"),
        ("1", PLAYER_UUID, "What is the price of that red powder?", "['3']", "False"),
        ("2", PLAYER_UUID, "Surely you can offer a better price to a friend.", "['4']", "Insight"),
        ("3", "npc-merchant", "Ten coppers a pinch, and worth every one.", "['no_links']", "False"),
        ("4", "npc-merchant", "A friend, is it? Fine, eight coppers, and not one less.", "['no_links']", "False"),
    ]
    for idx, spk, text, replies, skill in poe2:
        rows.append(dict(game="poe2", filename="toy/toy_merchant.conversation", id=idx, sid=float("nan"),
                         replies=replies, speaker=spk, persuasion=skill, en=text))
    return pd.DataFrame(rows)


def run(cmd, cwd, env=None, stdin=None, label=""):
    print(f"\n=== {label}: {' '.join(str(c) for c in cmd)}  (cwd={cwd})", flush=True)
    e = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8", **(env or {}))
    p = subprocess.run(cmd, cwd=cwd, env=e, input=stdin, text=True, capture_output=True,
                       encoding="utf-8", errors="replace")
    tail = (p.stdout + p.stderr).strip().splitlines()[-15:]
    print("\n".join("    " + t for t in tail))
    if p.returncode != 0:
        raise SystemExit(f"FAILED: {label} (exit {p.returncode})")
    print(f"--- OK: {label}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the temp directory")
    args = ap.parse_args()
    py = sys.executable
    tmp = Path(tempfile.mkdtemp(prefix="rpg_gpt_smoke_"))
    print("work dir:", tmp)
    try:
        # --- 1-2: dialogue paths (rpg_texts layout: data/..., data/data/to_concat_with_speakers/)
        texts = tmp / "rpg_texts"
        (texts / "data" / "data" / "to_concat_with_speakers").mkdir(parents=True)
        toy_dialogue_table().to_csv(texts / "data" / PERSUASION_CSV, sep="\t", encoding="utf-8")
        tools = REPO / "rpg_texts" / "extracting_data" / "tools"
        run([py, tools / "dialogpt_dataset_recursive.py"], texts, label="1 dialogue trees")
        run([py, tools / "concat_dialog_files.py"], texts, label="2 concat dialogue files")
        nlg = pd.read_csv(texts / "NLG_clean_with_speakers.csv", sep="\t")
        print(f"    NLG_clean_with_speakers.csv: {len(nlg)} dialogue paths")

        # --- 3-4: prompt/completion jsonl (rpg_bot_mahti layout: data/...)
        mahti = tmp / "rpg_bot_mahti"
        (mahti / "data").mkdir(parents=True)
        shutil.copy(texts / "NLG_clean_with_speakers.csv", mahti / "data")
        run([py, REPO / "rpg_bot_mahti" / "data" / "convert_to_json.py"], mahti, label="3 prompt/completion pairs")
        shutil.move(mahti / "NLG_RPG.jsonl", mahti / "data" / "NLG_RPG.jsonl")
        run([py, REPO / "rpg_bot_mahti" / "data" / "data_clean.py"], mahti, label="4 clean jsonl")
        pairs = [json.loads(l) for l in open(mahti / "data" / "NLG_RPG.jsonl", encoding="utf-8")]
        print(f"    NLG_RPG.jsonl: {len(pairs)} pairs; example: {pairs[0]}")
        assert pairs and all(p["completion"].startswith("NPC:") for p in pairs)

        # --- 5-6: GODEL fine-tuning + inference with a tiny random T5
        env = dict(RPG_GPT_BASE_MODEL=TINY_T5, RPG_GPT_EPOCHS="1", RPG_GPT_MAX_STEPS="2",
                   RPG_GPT_BATCH_SIZE="2", RPG_GPT_OUTPUT_DIR="results")
        run([py, REPO / "rpg_bot_mahti" / "godel_finetuning.py"], mahti, env=env, label="5 GODEL fine-tuning (tiny T5)")
        run([py, REPO / "rpg_bot_mahti" / "talk_w_godel.py"], mahti, env=dict(RPG_GPT_CHECKPOINT="results"),
            stdin="Who are you?\nTell me about the archive.\n", label="6 GODEL chat (tiny T5)")

        # --- 7: DialoGPT-style fine-tuning with a tiny GPT-2 (needs >= 4 context turns per row)
        base = ["Player: Hello there.", "NPC: Greetings, traveller.", "Player: Any work for me?",
                "NPC: Perhaps. The mill needs guarding.", "Player: How much does it pay?"]
        rows = [dict(zip(["response", "context/0", "context/1", "context/2", "context/3", "context/4"],
                         [f"NPC: Twenty coins, offer number {i}."] + [f"{t} ({i})" for t in reversed(base)]),
                     game="poe2") for i in range(20)]
        dlg = tmp / "dialogpt"
        (dlg / "data").mkdir(parents=True)
        pd.DataFrame(rows).to_csv(dlg / "data" / "nlg_data_10_turns_all_data.csv", sep="\t")
        env = dict(DIALOGPT_MODEL=TINY_GPT2, DIALOGPT_OUTPUT_DIR="output-tiny", DIALOGPT_EPOCHS="1",
                   DIALOGPT_MAX_STEPS="2")
        run([py, REPO / "rpg_texts" / "models" / "train_dialogpt.py"], dlg, env=env, label="7 DialoGPT fine-tuning (tiny GPT-2)")
        assert (mahti / "results" / "config.json").exists(), "GODEL model was not saved"
        assert (dlg / "output-tiny" / "config.json").exists(), "DialoGPT model was not saved"

        # --- 8: survey analysis on the anonymised responses
        ev = tmp / "eval"
        (ev / "evaluation").mkdir(parents=True)
        shutil.copy(REPO / "rpg_bot_mahti" / "evaluation" / "survey_data861_anonymised.csv", ev / "evaluation")
        for script in ["eval_explorationall.py", "eval_exploration.py", "pca.py"]:
            run([py, REPO / "rpg_bot_mahti" / "evaluation" / script], ev, label=f"8 evaluation/{script}")
        print("\nALL SMOKE TESTS PASSED")
    finally:
        if args.keep:
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
