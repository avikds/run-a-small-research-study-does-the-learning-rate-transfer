"""
Run a Small Research Study: Does the Learning Rate Transfer?

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_mlp
import math
import numpy as np
import torch
import torch.nn as nn

D_IN, D_OUT, BASE_WIDTH = 16, 4, 16


def load_task(n_train=2048, n_val=1024, seed=0):
    """Inputs and noisy outputs of a fixed random teacher network: (X_train, Y_train, X_val, Y_val)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_train + n_val, D_IN, generator=g)
    W1 = torch.randn(D_IN, 64, generator=g) / math.sqrt(D_IN)
    W2 = torch.randn(64, 64, generator=g) / math.sqrt(64)
    W3 = torch.randn(64, D_OUT, generator=g) / math.sqrt(64)
    Y = torch.tanh(torch.tanh(X @ W1 * 2) @ W2 * 2) @ W3
    Y = Y + 0.1 * torch.randn(Y.shape, generator=g)
    return X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]


def build_mlp(width, param, seed, base_width=BASE_WIDTH):
    """A 3-hidden-layer ReLU MLP initialized under the standard parametrization or muP."""

    # Validate the parametrization.
    if param not in ("sp", "mup"):
        raise ValueError("param must be either 'sp' or 'mup'.")

    # Set the random seed before creating the model layers.
    torch.manual_seed(seed)

    # Build the MLP with exactly seven modules.
    # All linear layers are bias-free.
    model = nn.Sequential(
        nn.Linear(D_IN, width, bias=False),
        nn.ReLU(),
        nn.Linear(width, width, bias=False),
        nn.ReLU(),
        nn.Linear(width, width, bias=False),
        nn.ReLU(),
        nn.Linear(width, D_OUT, bias=False)
    )

    # Collect the four linear layers in their forward-pass order.
    linear_layers = [
        layer for layer in model
        if isinstance(layer, nn.Linear)
    ]

    # Reinitialize weights according to the selected parametrization.
    with torch.no_grad():

        for i, layer in enumerate(linear_layers):

            # Input layer initialization.
            if i == 0:
                std = 1.0 / math.sqrt(D_IN)

            # Two hidden layers initialization.
            elif i in (1, 2):
                std = 1.0 / math.sqrt(width)

            # Final readout layer initialization.
            else:
                std = 1.0 / math.sqrt(width)

                # Apply the additional width-dependent scaling for muP.
                if param == "mup":
                    std *= math.sqrt(base_width / width)

            # Initialize weights from a zero-mean normal distribution.
            nn.init.normal_(
                layer.weight,
                mean=0.0,
                std=std
            )

    return model

# Step 2 - make_optimizer
def make_optimizer(model, lr, param, base_width=BASE_WIDTH):
    """Adam with the input layer in one group and the other weights, scaled for muP, in another."""

    # Validate the parametrization.
    if param not in ("sp", "mup"):
        raise ValueError("param must be either 'sp' or 'mup'.")

    # Obtain the hidden width from the input layer.
    width = model[0].out_features

    # Parameter group 1: input layer weights.
    # The input layer always uses the original learning rate.
    input_params = [model[0].weight]

    # Parameter group 2: weights of all remaining linear layers.
    other_params = [
        layer.weight
        for layer in model
        if isinstance(layer, nn.Linear) and layer is not model[0]
    ]

    # Calculate the learning rate for the remaining layers.
    if param == "mup":
        other_lr = lr * base_width / width
    else:
        other_lr = lr

    # Create Adam with exactly two parameter groups.
    # All other Adam settings retain their default values.
    optimizer = torch.optim.Adam(
        [
            {
                "params": input_params,
                "lr": lr
            },
            {
                "params": other_params,
                "lr": other_lr
            }
        ]
    )

    return optimizer

# Step 3 - coord_check
def coord_check(param, widths, data, lr=2 ** -10, steps=1, seed=0):
    """{width: RMS change of the output on X_val[:256] after `steps` Adam steps on X_train[:128]}."""

    # Unpack the training and validation datasets.
    X_train, Y_train, X_val, Y_val = data

    # Use fixed batches for every network width.
    X_batch = X_train[:128]
    Y_batch = Y_train[:128]
    X_probe = X_val[:256]

    # Store the RMS output changes for each width.
    results = {}

    # Evaluate each network width independently.
    for width in widths:

        # Build the model and its corresponding optimizer.
        model = build_mlp(width, param, seed)
        optimizer = make_optimizer(model, lr, param)

        # Record the initial output on the fixed probe batch.
        # Disable gradient tracking because this is only an evaluation.
        with torch.no_grad():
            initial_output = model(X_probe).clone()

        # Perform the specified number of Adam optimization steps.
        for _ in range(steps):

            # Clear gradients from the previous optimization step.
            optimizer.zero_grad()

            # Calculate predictions on the fixed training batch.
            predictions = model(X_batch)

            # Calculate mean squared error across all entries.
            loss = torch.mean((predictions - Y_batch) ** 2)

            # Compute gradients and update model parameters.
            loss.backward()
            optimizer.step()

        # Measure the output again after optimization.
        with torch.no_grad():
            final_output = model(X_probe)

        # Calculate the root mean square change across all
        # probe samples and output dimensions.
        rms_change = torch.sqrt(
            torch.mean((final_output - initial_output) ** 2)
        )

        # Store the result as a standard Python float.
        results[width] = float(rms_change.item())

    return results

# Step 4 - train_run
def train_run(width, param, lr, seed, data, steps=300, batch_size=128):
    """Train one model; return val_loss, per_example losses, diverged and steps_done."""

    # Unpack the training and validation datasets.
    X_train, Y_train, X_val, Y_val = data

    # Build the model and its optimizer.
    model = build_mlp(width, param, seed)
    optimizer = make_optimizer(model, lr, param)

    # Create a dedicated random generator for reproducible batch sampling.
    g = torch.Generator().manual_seed(1000 + seed)

    # Number of training examples.
    n_train = X_train.shape[0]

    # Track the number of successful optimizer updates.
    steps_done = 0

    # Training loop.
    for _ in range(steps):

        # Randomly sample a mini-batch using the dedicated generator.
        idx = torch.randint(
            0,
            n_train,
            (batch_size,),
            generator=g
        )

        X_batch = X_train[idx]
        Y_batch = Y_train[idx]

        # Clear gradients from the previous iteration.
        optimizer.zero_grad()

        # Forward pass.
        predictions = model(X_batch)

        # Calculate mean squared error across the batch and all outputs.
        loss = torch.mean((predictions - Y_batch) ** 2)

        # Check for divergence before performing the optimizer update.
        if not torch.isfinite(loss).item() or loss.item() > 100:

            return {
                "val_loss": None,
                "per_example": None,
                "diverged": True,
                "steps_done": steps_done
            }

        # Backpropagation.
        loss.backward()

        # Update model parameters.
        optimizer.step()

        # Increment the number of completed optimizer updates.
        steps_done += 1

    # Evaluate the trained model on the complete validation dataset.
    with torch.no_grad():

        val_predictions = model(X_val)

        # Calculate squared error averaged over the four output dimensions
        # for each individual validation example.
        per_example = (
            (val_predictions - Y_val) ** 2
        ).mean(dim=1).cpu().numpy()

    # Calculate the overall validation loss.
    val_loss = float(per_example.mean())

    # Check whether any validation loss is non-finite.
    if not np.all(np.isfinite(per_example)) or not np.isfinite(val_loss):

        return {
            "val_loss": None,
            "per_example": None,
            "diverged": True,
            "steps_done": steps
        }

    # Return the completed training results.
    return {
        "val_loss": val_loss,
        "per_example": per_example,
        "diverged": False,
        "steps_done": steps_done
    }

# Step 5 - lr_sweep
def lr_sweep(params, widths, lrs, seeds, data, steps=300, batch_size=128):
    """One row per (param, width, lr, seed) run, in that loop order."""

    # Store the results of all training runs.
    rows = []

    # Iterate over parametrizations first.
    for param in params:

        # Iterate over network widths.
        for width in widths:

            # Iterate over learning rates.
            for lr in lrs:

                # Iterate over random seeds (fastest-changing variable).
                for seed in seeds:

                    # Train one model with the current configuration.
                    result = train_run(
                        width=width,
                        param=param,
                        lr=lr,
                        seed=seed,
                        data=data,
                        steps=steps,
                        batch_size=batch_size
                    )

                    # Store the configuration and training results.
                    row = {
                        "param": param,
                        "width": width,
                        "lr": lr,
                        "seed": seed,
                        "val_loss": result["val_loss"],
                        "per_example": result["per_example"],
                        "diverged": result["diverged"]
                    }

                    # Append the current run to the results list.
                    rows.append(row)

    return rows

# Step 6 - summarize
def summarize(rows):
    """{(param, width): [one point per lr, sorted by lr]} with mean, std, n and diverged."""

    # Group experiment results by parametrization, width, and learning rate.
    grouped = {}

    for row in rows:
        key = (row["param"], row["width"])
        lr = row["lr"]

        # Create the nested dictionaries when necessary.
        if key not in grouped:
            grouped[key] = {}

        if lr not in grouped[key]:
            grouped[key][lr] = []

        # Store each run belonging to the current configuration.
        grouped[key][lr].append(row)

    # Store the summarized learning-rate curves.
    summary = {}

    # Process each parametrization and network width.
    for key, lr_groups in grouped.items():

        summary[key] = []

        # Sort learning rates in ascending order.
        for lr in sorted(lr_groups):

            runs = lr_groups[lr]

            # Count the total number of runs at this learning rate.
            n = len(runs)

            # Check whether any run diverged.
            diverged = any(run["diverged"] for run in runs)

            # If any run diverged, mark the entire point as diverged.
            if diverged:
                mean = None
                std = None

            else:
                # Collect validation losses from all runs.
                losses = [run["val_loss"] for run in runs]

                # Calculate the mean validation loss.
                mean = round(float(np.mean(losses)), 4)

                # Calculate sample standard deviation (ddof=1).
                # Standard deviation is None when fewer than 2 runs exist.
                if n >= 2:
                    std = round(float(np.std(losses, ddof=1)), 4)
                else:
                    std = None

            # Store the summarized result for this learning rate.
            point = {
                "lr": lr,
                "mean": mean,
                "std": std,
                "n": n,
                "diverged": diverged
            }

            summary[key].append(point)

    return summary

# Step 7 - best_lr
def best_lr(points):
    """The grid minimum and a parabola-vertex estimate of the best log2 learning rate."""

    # Handle the case where all learning-rate points diverged.
    valid_points = [
        point for point in points
        if not point["diverged"]
    ]

    if not valid_points:
        return {
            "grid_lr": None,
            "log2_lr": None,
            "status": "all diverged"
        }

    # Find the non-diverged point with the lowest mean validation loss.
    # Python's min() preserves the earliest point when values are tied.
    best_point = min(
        valid_points,
        key=lambda point: point["mean"]
    )

    # Record the best tested learning rate.
    grid_lr = best_point["lr"]

    # Find the position of the best point in the original curve.
    best_index = points.index(best_point)

    # Check whether the best point is at the boundary of the curve.
    if best_index == 0 or best_index == len(points) - 1:
        return {
            "grid_lr": grid_lr,
            "log2_lr": round(math.log2(grid_lr), 3),
            "status": "edge"
        }

    # Obtain the immediate left and right neighbours.
    left_point = points[best_index - 1]
    right_point = points[best_index + 1]

    # Avoid interpolation if either neighbouring point diverged.
    if left_point["diverged"] or right_point["diverged"]:
        return {
            "grid_lr": grid_lr,
            "log2_lr": round(math.log2(grid_lr), 3),
            "status": "edge"
        }

    # Prepare the three points for quadratic interpolation.
    # Use log2 learning rates as the x-coordinates.
    x = np.array([
        math.log2(left_point["lr"]),
        math.log2(best_point["lr"]),
        math.log2(right_point["lr"])
    ])

    # Use mean validation losses as the y-coordinates.
    y = np.array([
        left_point["mean"],
        best_point["mean"],
        right_point["mean"]
    ])

    # Fit a quadratic polynomial: y = ax^2 + bx + c.
    a, b, c = np.polyfit(x, y, 2)

    # Calculate the vertex of the fitted parabola.
    # The vertex estimates the optimal log2 learning rate.
    estimated_log2_lr = -b / (2 * a)

    # Return the grid minimum and interpolated estimate.
    return {
        "grid_lr": grid_lr,
        "log2_lr": round(float(estimated_log2_lr), 3),
        "status": "fit"
    }

# Step 8 - paired_penalty
def paired_penalty(runs_a, runs_b, n_boot=2000, seed=0):
    """Mean loss difference A - B, a paired bootstrap interval over examples, per-seed differences and a verdict."""

    # Convert the per-seed validation loss arrays into NumPy arrays.
    a = np.asarray(runs_a, dtype=float)
    b = np.asarray(runs_b, dtype=float)

    # Ensure both settings contain matching seed and example dimensions.
    if a.shape != b.shape:
        raise ValueError("runs_a and runs_b must have matching shapes.")

    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError(
            "runs_a and runs_b must have shape (seeds, examples) "
            "with non-empty dimensions."
        )

    if n_boot <= 0:
        raise ValueError("n_boot must be a positive integer.")

    # Calculate paired differences: validation loss of A minus B.
    # Shape: (number of seeds, number of validation examples).
    diff = a - b

    # Calculate the mean difference separately for each seed.
    per_seed = diff.mean(axis=1)

    # Average the per-example differences across all seeds.
    # This preserves the pairing of validation examples.
    d = diff.mean(axis=0)

    # Number of validation examples.
    n = d.shape[0]

    # Create a reproducible random generator for bootstrap sampling.
    rng = np.random.default_rng(seed)

    # Draw bootstrap samples by resampling validation examples
    # with replacement, using one shared index matrix.
    idx = rng.integers(
        0,
        n,
        size=(n_boot, n)
    )

    # Calculate the mean paired difference for each bootstrap sample.
    means = d[idx].mean(axis=1)

    # Calculate the 95% percentile bootstrap confidence interval.
    lo, hi = np.percentile(means, [2.5, 97.5])

    # Determine the verdict using both the confidence interval
    # and the direction of the differences across every seed.
    if lo > 0 and np.all(per_seed > 0):
        verdict = "a is worse"

    elif hi < 0 and np.all(per_seed < 0):
        verdict = "a is better"

    else:
        verdict = "can't tell"

    # Return all numerical results rounded to four decimal places.
    return {
        "diff": round(float(d.mean()), 4),
        "ci": (
            round(float(lo), 4),
            round(float(hi), 4)
        ),
        "per_seed": [
            round(float(value), 4)
            for value in per_seed
        ],
        "verdict": verdict
    }

# Step 9 - stability_edge
def stability_edge(points, best_log2):
    """The largest learning rate below the first divergence, and its distance from the best one."""

    # Sort the learning-rate points in ascending order.
    sorted_points = sorted(points, key=lambda point: point["lr"])

    # Track the last learning rate that completed without divergence.
    last_stable_lr = None

    # Walk through the learning-rate curve from smallest to largest.
    for point in sorted_points:

        # Stop at the first learning rate where training diverged.
        if point["diverged"]:
            break

        # Update the largest stable learning rate observed so far.
        last_stable_lr = point["lr"]

    # Handle the case where no stable learning rate was found.
    # This includes an empty curve or a curve whose first point diverged.
    if last_stable_lr is None:
        max_stable_log2 = None

    else:
        # Convert the largest stable learning rate to log2 space.
        max_stable_log2 = round(math.log2(last_stable_lr), 3)

    # Calculate the distance between the stability boundary
    # and the estimated optimal learning rate.
    if max_stable_log2 is None or best_log2 is None:
        headroom = None

    else:
        headroom = round(max_stable_log2 - best_log2, 3)

    # Return the stability boundary and available headroom.
    return {
        "max_stable_log2": max_stable_log2,
        "headroom": headroom
    }

# Step 10 - lr_width_fit
def lr_width_fit(widths, log2_lrs, new_width):
    """Slope and intercept of log2(best lr) against log2(width), and the prediction at new_width."""

    # Convert widths and optimal learning rates into NumPy arrays.
    widths = np.asarray(widths, dtype=float)
    log2_lrs = np.asarray(log2_lrs, dtype=float)

    # Ensure that the input arrays have matching lengths.
    if widths.ndim != 1 or log2_lrs.ndim != 1:
        raise ValueError("widths and log2_lrs must be one-dimensional.")

    if len(widths) != len(log2_lrs):
        raise ValueError("widths and log2_lrs must have the same length.")

    if len(widths) < 2:
        raise ValueError("At least two observations are required for fitting.")

    if np.any(widths <= 0) or new_width <= 0:
        raise ValueError("All widths must be positive.")

    # Transform network widths into log2 space.
    x = np.log2(widths)

    # Fit a least-squares linear model:
    # log2(best_lr) = intercept + slope * log2(width).
    slope, intercept = np.polyfit(x, log2_lrs, 1)

    # Predict the optimal log2 learning rate at the new width.
    predicted_log2_lr = intercept + slope * np.log2(new_width)

    # Return the fitted parameters and prediction.
    # Round all numerical results to three decimal places.
    return {
        "slope": round(float(slope), 3),
        "intercept": round(float(intercept), 3),
        "predicted_log2_lr": round(float(predicted_log2_lr), 3)
    }

# Step 11 - transfer_study
def transfer_study(rows, summary, small, big):
    """Per parametrization: the cost at width `big` of the learning rate tuned at width `small`."""

    # Collect all parametrizations and process them in sorted order.
    params = sorted({param for param, width in summary.keys()})

    # Store the transfer experiment results.
    results = {}

    for param in params:

        # Obtain the summarized learning-rate curves for both widths.
        small_curve = summary[(param, small)]
        big_curve = summary[(param, big)]

        # Find the best tested learning rate at the smaller width.
        tuned = best_lr(small_curve)["grid_lr"]

        # Find the independently optimized learning rate at the larger width.
        own = best_lr(big_curve)["grid_lr"]

        # Calculate the corresponding optimal rates in log2 space.
        # A missing optimum is represented by None.
        entry = {
            "tuned_log2": (
                round(math.log2(tuned), 3)
                if tuned is not None else None
            ),
            "big_best_log2": (
                round(math.log2(own), 3)
                if own is not None else None
            )
        }

        # A transfer experiment cannot be performed without
        # a valid learning rate tuned at the smaller width.
        if tuned is None:
            raise ValueError(
                f"No stable learning rate found at width {small} "
                f"for parametrization {param!r}."
            )

        # Locate the point corresponding to the transferred learning rate
        # in the larger network's summarized learning-rate curve.
        tuned_point = next(
            (
                point for point in big_curve
                if point["lr"] == tuned
            ),
            None
        )

        if tuned_point is None:
            raise ValueError(
                f"Learning rate {tuned} is missing from the "
                f"width {big} curve for parametrization {param!r}."
            )

        # Case 1: The transferred learning rate causes divergence.
        if tuned_point["diverged"]:

            entry["diverged"] = True
            entry["penalty"] = None

        # Case 2: The transferred learning rate is already
        # the larger network's own best grid learning rate.
        elif tuned == own:

            entry["diverged"] = False

            # No performance difference exists when both settings
            # use exactly the same learning rate.
            entry["penalty"] = {
                "diff": 0.0,
                "ci": (0.0, 0.0),
                "per_seed": [],
                "verdict": "same lr"
            }

        # Case 3: The transferred and independently optimized
        # learning rates are different.
        else:

            # Collect the large-width runs using the transferred
            # learning rate, sorted by increasing random seed.
            runs_a = sorted(
                [
                    row for row in rows
                    if row["param"] == param
                    and row["width"] == big
                    and row["lr"] == tuned
                ],
                key=lambda row: row["seed"]
            )

            # Collect the large-width runs using its own
            # optimal grid learning rate.
            runs_b = sorted(
                [
                    row for row in rows
                    if row["param"] == param
                    and row["width"] == big
                    and row["lr"] == own
                ],
                key=lambda row: row["seed"]
            )

            # Ensure that both configurations contain matching seeds.
            seeds_a = [row["seed"] for row in runs_a]
            seeds_b = [row["seed"] for row in runs_b]

            if seeds_a != seeds_b:
                raise ValueError(
                    "Transferred and optimal learning-rate runs "
                    "must contain matching seeds."
                )

            # Extract the per-example validation losses for each seed.
            a = [row["per_example"] for row in runs_a]
            b = [row["per_example"] for row in runs_b]

            # Calculate the paired validation-loss penalty.
            penalty = paired_penalty(a, b)

            entry["diverged"] = False
            entry["penalty"] = penalty

        # Store the result for the current parametrization.
        results[param] = entry

    return results

# Step 12 - write_conclusion
NAMES = {
    "sp": "Standard parametrization",
    "mup": "muP"
}


def write_conclusion(study, slopes, small, big):
    """The claims the evidence supports, one string per line."""

    # Store the final research conclusions.
    conclusions = []

    # Track whether each parametrization successfully transfers.
    transfer_status = {}

    # Generate one conclusion for each parametrization in sorted order.
    for param in sorted(study):

        name = NAMES.get(param, param)
        entry = study[param]

        diverged = entry["diverged"]
        penalty = entry["penalty"]

        verdict = penalty["verdict"] if penalty is not None else None

        # Case 1: The transferred learning rate causes divergence.
        if diverged:

            conclusions.append(
                f"{name}: the learning rate tuned at width {small} "
                f"diverges at width {big}."
            )

            transfer_status[param] = False

        # Case 2: Both widths have the same best learning rate.
        elif verdict == "same lr":

            conclusions.append(
                f"{name}: width {big} has the same best learning rate "
                f"as width {small}."
            )

            transfer_status[param] = True

        # Case 3: The transferred learning rate has a significantly
        # higher validation loss.
        elif verdict == "a is worse":

            diff = penalty["diff"]
            lo, hi = penalty["ci"]
            k = len(penalty["per_seed"])

            conclusions.append(
                f"{name}: the learning rate tuned at width {small} "
                f"costs {diff:+.4f} validation loss at width {big} "
                f"(95% CI {lo:.4f} to {hi:.4f}, "
                f"worse on all {k} seeds)."
            )

            transfer_status[param] = False

        # Case 4: No measurable transfer penalty was established.
        else:

            diff = penalty["diff"]
            lo, hi = penalty["ci"]

            conclusions.append(
                f"{name}: no measurable cost from the learning rate "
                f"tuned at width {small} "
                f"({diff:+.4f}, 95% CI {lo:.4f} to {hi:.4f})."
            )

            transfer_status[param] = True

    # Report the fitted learning-rate scaling slopes.
    for param in sorted(slopes):

        name = NAMES.get(param, param)
        slope = slopes[param]

        conclusions.append(
            f"{name}: best log2 learning rate changes by "
            f"{slope:+.2f} per doubling of width."
        )

    # Check whether both parametrizations were evaluated.
    both_available = (
        "mup" in transfer_status and "sp" in transfer_status
    )

    mup_transfers = transfer_status.get("mup", False)
    sp_transfers = transfer_status.get("sp", False)

    # Generate the overall research conclusion.
    if both_available and mup_transfers and not sp_transfers:

        final_conclusion = (
            "Conclusion: on this task the learning rate transfers "
            "from small to large width under muP and not under "
            "the standard parametrization."
        )

    elif both_available and mup_transfers and sp_transfers:

        final_conclusion = (
            "Conclusion: the learning rate transfers under both "
            "parametrizations here, so this task does not separate them."
        )

    else:

        final_conclusion = (
            "Conclusion: inconclusive; muP did not transfer "
            "the learning rate on this task."
        )

    # Append the overall conclusion as the final line.
    conclusions.append(final_conclusion)

    return conclusions

