# train.py

CACHE_DIR = "F:/hf_home"
MODEL_NAME = "Quokka_230M"
DATASET_SIZE = 200_000  # either an integer or "ALL" (to train on the entire dataset)
PERCENT_TRAIN = 1.0

from quokka.optimizers import SophiaG
from quokka.model import Quokka
from quokka.trainer import Trainer
from quokka.config import Quokka230MConfig, PretrainConfig
from quokka.data import get_dataloaders
import torch
import math

train_dataloader, val_dataloader, tokenizer = get_dataloaders(
  CACHE_DIR,
  DATASET_SIZE,
  PERCENT_TRAIN,
  openai_encoding="p50k_base"
)

quokka_config = Quokka230MConfig(vocab_size=tokenizer.n_vocab, max_seq_len=PretrainConfig.max_ctx_len, layer_dropout=0.0, feedforward_dropout=0.0)
model = Quokka(quokka_config, pretrained=False)
# checkpoint = torch.load('Quokka_230M_epoch2.pth')
# model.load_state_dict(checkpoint['model'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = SophiaG(
    model.parameters(),
    lr=PretrainConfig.learning_rate,
    betas=PretrainConfig.betas,
    rho=PretrainConfig.rho,
    weight_decay=PretrainConfig.weight_decay
)
# optimizer.load_state_dict(checkpoint['optimizer'])

# Dynamic -> (converted to) fixed number of steps per epoch
steps_per_epoch = sum(1 for _ in train_dataloader)
PretrainConfig.total_steps = PretrainConfig.num_epochs * math.ceil(steps_per_epoch / PretrainConfig.accumulation_steps)
warmup_steps = int(0.1 * PretrainConfig.total_steps)

from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=PretrainConfig.total_steps
)
# scheduler.load_state_dict(checkpoint['scheduler'])

# Mixed-precision scaler
scaler = torch.cuda.amp.GradScaler()

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    device=device,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader if PERCENT_TRAIN < 1.0 else None,
    tokenizer=tokenizer
)

# start training
trainer.run(MODEL_NAME)