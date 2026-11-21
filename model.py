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

# Step 3 - coord_check (not yet solved)
# TODO: implement

# Step 4 - train_run (not yet solved)
# TODO: implement

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

