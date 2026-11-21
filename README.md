# Run a Small Research Study: Does the Learning Rate Transfer?

Run a complete research study on a question every large-model team depends on: if you tune the learning rate on a narrow network, does it still work on a wide one? Build the same network under the standard parametrization and under muP, check the implementation with a coordinate check, and make every run reproducible. Sweep learning rates at three widths over three seeds, estimate each best rate between grid points, and find where training diverges. Then test the transfer with a paired bootstrap and a check across seeds, fit how the best rate moves with width, and write down only the claims the evidence supports.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** build_mlp
- [x] **2.** make_optimizer
- [x] **3.** coord_check
- [x] **4.** train_run
- [x] **5.** lr_sweep
- [x] **6.** summarize
- [x] **7.** best_lr
- [x] **8.** paired_penalty
- [x] **9.** stability_edge
- [x] **10.** lr_width_fit
- [x] **11.** transfer_study
- [x] **12.** write_conclusion

## Results

```
coordinate check sp {64: 0.0697, 256: 0.2478, 1024: 3.1862} | width 1024 / width 64: 45.7
coordinate check mup {64: 0.0091, 256: 0.0043, 1024: 0.0056} | width 1024 / width 64: 0.6
runs: 144 | diverged: 6 | seconds: 42
 sp width  16 | best log2 lr -7.112 (fit) | largest stable log2 lr -4.0 | headroom 3.112 | 0.3657 0.2476 0.2046 0.1888 0.1857 0.1906 0.1986 0.2078
 sp width  64 | best log2 lr -9.295 (fit) | largest stable log2 lr -4.0 | headroom 5.295 | 0.1971 0.1893 0.1885 0.1916 0.1981 0.1981 0.2023 0.2653
 sp width 256 | best log2 lr -10.048 (fit) | largest stable log2 lr -6.0 | headroom 4.048 | 0.1679 0.1651 0.1685 0.1783 0.1827 0.1947 div div
 sp fit: log2 lr* = -4.414 + -0.734 log2 width | predicted at width 1024: -11.754
mup width  16 | best log2 lr -7.112 (fit) | largest stable log2 lr -4.0 | headroom 3.112 | 0.3657 0.2476 0.2046 0.1888 0.1857 0.1906 0.1986 0.2078
mup width  64 | best log2 lr -8.297 (fit) | largest stable log2 lr -4.0 | headroom 4.297 | 0.2088 0.1879 0.1808 0.1793 0.1852 0.1968 0.1959 0.2034
mup width 256 | best log2 lr -6.3 (fit) | largest stable log2 lr -4.0 | headroom 2.3 | 0.1782 0.1745 0.1727 0.1687 0.1611 0.1590 0.1674 0.1725
mup fit: log2 lr* = -8.454 + 0.203 log2 width | predicted at width 1024: -6.424
transfer mup {'tuned_log2': -7.0, 'big_best_log2': -6.0, 'diverged': False, 'penalty': {'diff': 0.0021, 'ci': (-0.001, 0.0051), 'per_seed': [0.0031, -0.0014, 0.0045], 'verdict': "can't tell"}}
transfer sp {'tuned_log2': -7.0, 'big_best_log2': -10.0, 'diverged': False, 'penalty': {'diff': 0.0176, 'ci': (0.0133, 0.022), 'per_seed': [0.0099, 0.0253, 0.0176], 'verdict': 'a is worse'}}

muP: no measurable cost from the learning rate tuned at width 16 (+0.0021, 95% CI -0.0010 to 0.0051).
Standard parametrization: the learning rate tuned at width 16 costs +0.0176 validation loss at width 256 (95% CI 0.0133 to 0.0220, worse on all 3 seeds).
muP: best log2 learning rate changes by +0.20 per doubling of width.
Standard parametrization: best log2 learning rate changes by -0.73 per doubling of width.
Conclusion: on this task the learning rate transfers from small to large width under muP and not under the standard parametrization.
```
