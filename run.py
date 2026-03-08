from pathlib import Path
import os

from litgpt.finetune.full_sens_quant12 import setup
from litgpt.args import TrainArgs, EvalArgs, LogArgs
from litgpt.data import Alpaca


def find_repo_root(start: Path) -> Path:
    """Find repo root by walking up until we see 'checkpoints/' (adjust if your layout differs)."""
    p = start
    for _ in range(10):
        if (p / "checkpoints").exists():
            return p
        if p == p.parent:
            break
        p = p.parent
    # fallback: parent of this file
    return start.parent


def main():
    here = Path(__file__).resolve()
    repo = find_repo_root(here.parent)

    checkpoint_dir = (repo / "checkpoints" / "meta-llama" / "Llama-3.2-1B").resolve()

    out_dir = (repo / "out" / "full_sens_quant4" / "bs128_ms3000_seq256").resolve()

    setup(
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        precision=None,

        devices=1,
        num_nodes=1,
        resume=False,
        data=Alpaca(num_workers=0),

        train=TrainArgs(
            max_steps=3000,
            epochs=30,
            global_batch_size=128,
            micro_batch_size=4,
            lr_warmup_steps=100,


            log_interval=320,
            save_interval=None,

            max_seq_length=256,
        ),

        
        eval=EvalArgs(
            interval=100,
            max_iters=100,
            initial_validation=True,
            final_validation=True,
            max_new_tokens=50,
        ),

        log=LogArgs(),

        optimizer="AdamW",

        logger_name="csv",
        seed=1337,

        access_token=os.environ.get("HF_TOKEN", None),
    )


if __name__ == "__main__":
    main()
