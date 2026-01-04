import argparse
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer


def read_dir(path: Path) -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8").strip() for p in path.glob("*.txt")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    pred_dir = Path(args.pred_dir)

    refs = read_dir(ref_dir)
    preds = read_dir(pred_dir)

    common_ids = sorted(refs.keys() & preds.keys())
    if not common_ids:
        raise RuntimeError("No matching UtteranceID between ref_dir and pred_dir")

    cer_sum = 0.0
    wer_sum = 0.0

    for uid in common_ids:
        cer_sum += calc_cer(refs[uid], preds[uid])
        wer_sum += calc_wer(refs[uid], preds[uid])

    n = len(common_ids)
    print(f"Samples: {n}")
    print(f"CER: {cer_sum / n:.6f}")
    print(f"WER: {wer_sum / n:.6f}")


if __name__ == "__main__":
    main()
