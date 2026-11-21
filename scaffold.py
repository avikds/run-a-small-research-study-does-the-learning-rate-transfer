"""
Run a Small Research Study: Does the Learning Rate Transfer? scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Run a small research study: does a learning rate tuned on a narrow network still work on a wide one?

Story: build the same MLP under the standard parametrization and under muP, check the implementation with
a coordinate check, sweep learning rates at three widths over three seeds, find each best learning rate,
then test whether the narrow model's best rate costs anything at the widest width, with a paired bootstrap
over validation examples and a check across seeds. Finish with only the claims the evidence supports.
"""
import time

SMALL, MID, BIG = 16, 64, 256
LRS = [2.0 ** k for k in range(-11, -3)]
SEEDS = [0, 1, 2]


def main():
    t0 = time.time()
    data = load_task()

    # ---- Check the implementation before spending compute ----
    for param in ("sp", "mup"):
        change = coord_check(param, [64, 256, 1024], data)
        print("coordinate check", param, {w: round(v, 4) for w, v in change.items()},
              "| width 1024 / width 64:", round(change[1024] / change[64], 1))

    # ---- The sweep: 2 parametrizations x 3 widths x 8 learning rates x 3 seeds ----
    rows = lr_sweep(["sp", "mup"], [SMALL, MID, BIG], LRS, SEEDS, data)
    summary = summarize(rows)
    print("runs:", len(rows), "| diverged:", sum(r["diverged"] for r in rows), "| seconds:", round(time.time() - t0))

    slopes, best = {}, {}
    for param in ("sp", "mup"):
        logs = []
        for width in (SMALL, MID, BIG):
            points = summary[(param, width)]
            b = best_lr(points)
            edge = stability_edge(points, b["log2_lr"])
            logs.append(b["log2_lr"])
            curve = " ".join("div" if p["diverged"] else f"{p['mean']:.4f}" for p in points)
            print(f"{param:>3} width {width:>3} | best log2 lr {b['log2_lr']} ({b['status']}) | "
                  f"largest stable log2 lr {edge['max_stable_log2']} | headroom {edge['headroom']} | {curve}")
        fit = lr_width_fit([SMALL, MID, BIG], logs, 1024)
        slopes[param] = fit["slope"]
        best[param] = fit
        print(f"{param:>3} fit: log2 lr* = {fit['intercept']} + {fit['slope']} log2 width | predicted at width 1024: {fit['predicted_log2_lr']}")

    study = transfer_study(rows, summary, SMALL, BIG)
    for param, e in study.items():
        print("transfer", param, e)
    print()
    for line in write_conclusion(study, slopes, SMALL, BIG):
        print(line)


if __name__ == "__main__":
    main()

