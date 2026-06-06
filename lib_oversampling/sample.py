import sys
sys.path.append('../')
import os
from ddpm import modules, diffusion, train
import torch
import data_utils as du
from ddpm.resample import create_named_schedule_sampler


def resolve_binary_target_index(label_wrapper, label_column, target):
    values = list(label_wrapper.all_distinct_values.get(label_column, []))
    if len(values) != 2:
        raise ValueError(
            f"Original RelDDPM requires exactly two values for label column "
            f"'{label_column}', found {len(values)}."
        )

    target_text = str(target).strip()
    matches = [i for i, value in enumerate(values) if str(value).strip() == target_text]
    if not matches:
        try:
            target_number = float(target_text)
        except (TypeError, ValueError):
            target_number = None
        if target_number is not None:
            for i, value in enumerate(values):
                try:
                    if float(value) == target_number:
                        matches.append(i)
                except (TypeError, ValueError):
                    continue

    if len(matches) != 1:
        raise ValueError(
            f"Target value {target!r} is not a unique value in label column "
            f"'{label_column}'; expected one of {[str(v) for v in values]}."
        )
    return float(matches[0])


def get_cond_fn(controller, scale_factor, label, n_classes=2):
    
    def cond_fn(c, x, t):
        x = x.float()
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = controller(x_in, t)
            if n_classes == 2:
                if label == 1.0:
                    gradients = torch.autograd.grad(logits.sum(), x_in)[0] * scale_factor
                elif label == 0.0:
                    gradients = -torch.autograd.grad(logits.sum(), x_in)[0] * scale_factor
            return gradients

    return cond_fn


def oversampling(n_samples, controller, diffuser, label, device, n_classes=2, scale_factor=8.0):
    controller.to(device)
    diffuser.to(device)
    diffuser.variables_to_device(device)

    cond_fn = get_cond_fn(controller, scale_factor, label, n_classes)
    cond = torch.zeros(n_samples, 1)

    samples = diffuser.sample(n_samples, control_tools=[cond, cond_fn])
    return samples
