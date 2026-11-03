"""Correct Transformers 4.51.3's floor-vs-remainder epoch accounting.

The pinned Trainer flushes a final partial accumulation group but computes the
epoch-based stopping step using floor division. Use ceil in that accounting so
every example is visited in each requested complete epoch, including on resume.
"""
import math

from transformers import Trainer


class ExactEpochTrainer(Trainer):
    def set_initial_training_values(self, args, dataloader, total_train_batch_size):
        values = list(super().set_initial_training_values(args, dataloader, total_train_batch_size))
        # Tuple contract in the pinned version: epochs, updates/epoch, examples,
        # samples, epoch_based, dataloader_length, max_steps.
        if values[4] and values[5] is not None:
            values[1] = max(math.ceil(values[5] / args.gradient_accumulation_steps), 1)
            values[6] = math.ceil(args.num_train_epochs * values[1])
        return tuple(values)
