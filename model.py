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

# Step 5 - lr_sweep (not yet solved)
# TODO: implement

# Step 6 - summarize (not yet solved)
# TODO: implement

# Step 7 - best_lr (not yet solved)
# TODO: implement

# Step 8 - paired_penalty (not yet solved)
# TODO: implement

# Step 9 - stability_edge (not yet solved)
# TODO: implement

# Step 10 - lr_width_fit (not yet solved)
# TODO: implement

# Step 11 - transfer_study (not yet solved)
# TODO: implement

# Step 12 - write_conclusion (not yet solved)
# TODO: implement

