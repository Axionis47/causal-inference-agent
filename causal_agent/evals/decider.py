"""Fine-tune the decision model on the labelled turns, then score it on the held-out ones:

    uv run python -m causal_agent.evals.decider                       # the open English checkpoint, tuned into .artifacts/decider/turns
    uv run python -m causal_agent.evals.decider --out DIR --epochs 6  # elsewhere, longer
    uv run python -m causal_agent.evals.decider --freeze-encoder --max-len 256 --loss soft-ce   # the head alone: minutes on a laptop

The published recipe (the whole encoder, the RLCD loss, 512 tokens) is the default and wants a GPU: on an Apple laptop it did not
finish one epoch in two hours. The frozen-encoder run is the laptop's, and on 2026-10-06 it learned nothing: chance accuracy on the
calibration items and 23% on the held-out turns at threshold zero against the open checkpoint's 25%, so no checkpoint is in use
and `desk/decider.yaml` keeps the threshold the open checkpoint earns. A fine-tune replaces it only when this sweep says so.

The training rows are `desk/evals/turns_train.jsonl` rendered by `evals/turns.py::to_laya`, so the model trains on exactly the
state and the questions the desk sends. The result loads with `DECIDER_CHECKPOINT=<out>`; nothing in the desk changes. The
held-out score printed at the end is what decides whether the fine-tune replaces the open checkpoint, and at which threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_agent.common import config, decider, llm
from causal_agent.evals import turns


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.get().paths.root / ".artifacts" / "decider" / "turns"))
    ap.add_argument("--base", default="english", help="the checkpoint to start from: a Laya name or a local directory")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--freeze-encoder", action="store_true", help="train the decision head only; minutes instead of hours")
    ap.add_argument("--max-len", type=int, default=None, help="tokens of state the model reads; the desk's states are short")
    ap.add_argument("--loss", default="rlcd", choices=["rlcd", "soft-ce"])
    ap.add_argument("--train-set", default=str(turns.TRAIN_SET))
    ap.add_argument("--set", default=str(turns.SET), help="the held-out turns scored afterwards")
    args = ap.parse_args(argv)

    from laya.train import TrainConfig, finetune, resolve_checkpoint_dir

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = out / "train.jsonl"
    rows = turns.rows_of(Path(args.train_set))
    data.write_text("".join(json.dumps(turns.to_laya(r)) + "\n" for r in rows))
    print(f"{len(rows)} labelled turns -> {data}")
    # options shuffled per epoch so the kind is learned from the words, not from the slot it sits in; a fifth of the items
    # held back to fit the confidence, since a threshold on an uncalibrated confidence means nothing
    cfg = TrainConfig(
        epochs=args.epochs,
        micro_batch=8,
        grad_accum=2,
        shuffle_options=("choice",),
        calib_frac=0.2,
        calib_max=400,
        freeze_encoder=args.freeze_encoder,
        max_len=args.max_len,
        loss=args.loss,
        log_every=10,
    )
    summary = finetune(str(data), resolve_checkpoint_dir(args.base), str(out), cfg)
    print("epoch loss", [round(x, 4) for x in summary["epoch_loss"]], "temperature", summary["temperature"])
    for name, entry in summary["calibration"].items():
        print(f"calibration {name}: {entry}")

    llm.set_decider(decider.Laya(checkpoint=str(out)))
    turns.main(["--set", args.set])
    print(f"\nto use it: DECIDER_CHECKPOINT={out}")


if __name__ == "__main__":
    main()
