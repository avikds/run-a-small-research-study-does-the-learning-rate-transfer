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
- [ ] **6.** summarize
- [ ] **7.** best_lr
- [ ] **8.** paired_penalty
- [ ] **9.** stability_edge
- [ ] **10.** lr_width_fit
- [ ] **11.** transfer_study
- [ ] **12.** write_conclusion

---

Built on Deep-ML.
