# src/quokka/trainer.py

import torch
import math
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from quokka.utils import clear_memory
from quokka.data import collate_fn, FilteredIterableDataset, DataLoader
from quokka.config import PretrainConfig

def calculate_perplexity(loss):
    return math.exp(loss)

class Trainer:
    def __init__(self, model, optimizer, scheduler, scaler, device, train_dataloader, val_dataloader, tokenizer):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        self.train_losses = []
        self.val_losses = []
        self.train_perplexities = []
        self.val_perplexities = []
        self.total_epochs = 0
        self.global_step = 0
        self.perplexities = {}  # dict stores perplexities per data block

    def train_epoch(self, dataloader, accumulation_steps, max_grad_norm, epoch_desc, record_perplexities=False):
        total_train_loss = 0.0
        total_train_tokens = 0

        self.model.train()
        loop = tqdm(enumerate(dataloader), total=sum(1 for _ in dataloader), leave=True)
        self.optimizer.zero_grad()
        for step, batch in loop:
            inputs = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            data_block_ids = batch.get('ids', None)  # in case ids are needed

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, loss = self.model(inputs, targets=labels, return_loss=True)

            if record_perplexities and data_block_ids is not None:
                with torch.no_grad():
                    logits_flat = logits.view(-1, logits.size(-1))
                    labels_flat = labels.view(-1)
                    losses_flat = nn.functional.cross_entropy(
                        logits_flat,
                        labels_flat,
                        reduction='none',
                        ignore_index=-1
                    )
                    losses = losses_flat.view(inputs.size(0), -1).sum(dim=1) / (labels != -1).sum(dim=1)
                    batch_perplexities = torch.exp(losses)

                    for idx, perplexity_value in zip(data_block_ids, batch_perplexities):
                        if idx in self.perplexities:
                            self.perplexities[idx].append(perplexity_value.item())
                        else:
                            self.perplexities[idx] = [perplexity_value.item()]

            self.scaler.scale(loss).backward()

            if (self.global_step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            if (self.global_step + 1) % accumulation_steps == 0:
                loop.set_description(epoch_desc)
                loop.set_postfix(loss=loss.item())

            # accumulate total loss and total tokens
            batch_loss = loss.item() * inputs.numel()
            total_train_loss += batch_loss
            total_train_tokens += inputs.numel()

            self.global_step += 1

        # average training loss per token and perplexity
        avg_train_loss = total_train_loss / total_train_tokens
        perplexity = calculate_perplexity(avg_train_loss)

        return avg_train_loss, perplexity

    def perform_validation(self):
        # (optional) perform validation
        def validation_step():
            self.model.eval()
            total_val_loss = 0.0
            total_val_tokens = 0

            with torch.no_grad():
                for step, batch in enumerate(self.val_dataloader):
                    inputs = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        logits, loss = self.model(inputs, targets=labels, return_loss=True)

                    # accumulate total loss and total tokens
                    batch_loss = loss.item() * inputs.numel()
                    total_val_loss += batch_loss
                    total_val_tokens += inputs.numel()

            # calculate average validation loss and perplexity
            avg_val_loss = (total_val_loss / total_val_tokens) if total_val_tokens > 0 else None
            perplexity = calculate_perplexity(avg_val_loss) if avg_val_loss is not None else None
            return avg_val_loss, perplexity

        if self.val_dataloader:
            avg_val_loss, val_perp = validation_step()
            self.val_losses.append(avg_val_loss)
            self.val_perplexities.append(val_perp)

    def plot_results(self, MODEL_NAME):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # plot Train Perplexity on the first subplot
        epochs = range(1, len(self.train_perplexities) + 1)
        axs[0].plot(epochs, self.train_perplexities, label="Train Perplexity", marker="o")
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("Perplexity")
        axs[0].set_title("Train Perplexity over Epochs")
        axs[0].legend()

        if len(self.val_perplexities) > 0:
            axs[1].plot(epochs[:len(self.val_perplexities)], self.val_perplexities, label="Validation Perplexity", marker="o")
            axs[1].set_xlabel("Epochs")
            axs[1].set_ylabel("Perplexity")
            axs[1].set_title("Validation Perplexity over Epochs")
            axs[1].legend()
        else:
            # hide the second subplot if there are no validation perplexities
            fig.delaxes(axs[1])

        # adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(f"{MODEL_NAME}_epoch{PretrainConfig.num_epochs}.png")
        plt.show()

    def save_model(self, MODEL_NAME):
        print("Training complete. Saving final model...")
        checkpoint = {
            'epoch': PretrainConfig.num_epochs,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(checkpoint, f'{MODEL_NAME}_epoch{PretrainConfig.num_epochs}.pth')

    def run(self, MODEL_NAME):
        accumulation_steps = PretrainConfig.accumulation_steps
        max_grad_norm = PretrainConfig.max_grad_norm

        # Phase 1: Learn phase
        print("Starting Phase 1: Learn phase...")
        for _ in range(PretrainConfig.p1):
            avg_train_loss, perplexity = self.train_epoch(
                dataloader=self.train_dataloader,
                accumulation_steps=accumulation_steps,
                max_grad_norm=max_grad_norm,
                epoch_desc=f'Epoch [{self.total_epochs+1}/{PretrainConfig.num_epochs}]',
                record_perplexities=True
            )
            self.train_losses.append(avg_train_loss)
            self.train_perplexities.append(perplexity)
            self.total_epochs += 1

        # average perplexity per data block
        avg_perplexities = {idx: sum(ppls)/len(ppls) for idx, ppls in self.perplexities.items()}

        # sort data blocks by average perplexity in ascending order (easiest to hardest)
        sorted_perplexities = sorted(avg_perplexities.items(), key=lambda x: x[1])

        # compute the number of data blocks to discard
        s1 = PretrainConfig.s1
        num_blocks = len(sorted_perplexities)
        num_blocks_to_discard = int(num_blocks * s1 / 100)

        # get the indices of the hardest data blocks
        # i.e. what you struggle with the most when preparing for exams
        hard_data_block_ids = [idx for idx, _ in sorted_perplexities[num_blocks_to_discard:]]

        # Phase 2: Focus phase
        print("Starting Phase 2: Focus phase...")
        focus_dataset = FilteredIterableDataset(self.train_dataloader.dataset, hard_data_block_ids)
        focus_dataloader = DataLoader(focus_dataset, batch_size=PretrainConfig.batch_size, collate_fn=collate_fn)

        for _ in range(PretrainConfig.p2):
            avg_train_loss, perplexity = self.train_epoch(
                dataloader=focus_dataloader,
                accumulation_steps=accumulation_steps,
                max_grad_norm=max_grad_norm,
                epoch_desc=f'Focus Epoch [{self.total_epochs+1}/{PretrainConfig.num_epochs}]',
                record_perplexities=False
            )
            self.total_epochs += 1

        # Phase 3: Review phase
        print("Starting Phase 3: Review phase...")
        # Empty perplexities dict
        self.perplexities = {}
        clear_memory()

        for _ in range(PretrainConfig.p3):
            avg_train_loss, perplexity = self.train_epoch(
                dataloader=self.train_dataloader,
                accumulation_steps=accumulation_steps,
                max_grad_norm=max_grad_norm,
                epoch_desc=f'Epoch [{self.total_epochs+1}/{PretrainConfig.num_epochs}]',
                record_perplexities=True
            )
            self.train_losses.append(avg_train_loss)
            self.train_perplexities.append(perplexity)
            self.total_epochs += 1

        # similar calculations as Phase 3
        avg_perplexities = {idx: sum(ppls)/len(ppls) for idx, ppls in self.perplexities.items()}
        sorted_perplexities = sorted(avg_perplexities.items(), key=lambda x: x[1])

        s2 = PretrainConfig.s2
        num_blocks = len(sorted_perplexities)
        num_blocks_to_discard = int(num_blocks * s2 / 100)

        hard_data_block_ids = [idx for idx, _ in sorted_perplexities[num_blocks_to_discard:]]

        clear_memory()

        # Phase 4: Focus phase repeated reps times
        for rep in range(PretrainConfig.reps):
            focus_dataset = FilteredIterableDataset(self.train_dataloader.dataset, hard_data_block_ids)
            focus_dataloader = DataLoader(focus_dataset, batch_size=PretrainConfig.batch_size, collate_fn=collate_fn)

            for _ in range(PretrainConfig.p4):
                avg_train_loss, perplexity = self.train_epoch(
                    dataloader=focus_dataloader,
                    accumulation_steps=accumulation_steps,
                    max_grad_norm=max_grad_norm,
                    epoch_desc=f'Phase 4 Rep {rep+1} Epoch [{self.total_epochs+1}/{PretrainConfig.num_epochs}]',
                    record_perplexities=False
                )
                self.train_losses.append(avg_train_loss)
                self.train_perplexities.append(perplexity)
                self.total_epochs += 1

        self.perform_validation()
        self.plot_results(MODEL_NAME)
        self.save_model(MODEL_NAME)