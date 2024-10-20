# <a href="https://github.com/aistrova/quokka" target="_blank"><img src="./assets/quokka-llm.png" width="60"></a> [Quokka](https://github.com/aistrova/quokka) : Easiest Way to Create, Train, and Customize SoTA LLMs

Quokka is the one and only fully open-source SoTA LLM framework for AI education. All code and features are optimized for exploring and understanding the deep learning behind LLMs like ChatGPT.


## Educational Features

1. **3x faster download**
   - download only the necessary rows of data as you train.
   - instead of downloading 100+ GB of data, stream a few KB at a time (often 3x faster than GET).

2. **Infinite dataset shuffling**
   - tokenize the training data on the fly.
   - instead of loading all training data into memory, load one batch at a time.
   - dataset loaders with 10K and 100M rows of text use the same RAM, even with shuffling enabled! 🤯

3. **Efficient long-context pretraining**
   - instead of traditional training using one row of text per forward pass, fill up the tokens to the brim,
   - thus significantly reducing the number of steps per epoch.

4. **Least overwhelming**
   - Quokka is more complete, easier to understand, and easier to build/train new SoTA models compared to existing open-source Llama 3.1, GPT-3, and Transformer implementations.

5. **Extremely simple**
   - the main Quokka model is just **157 lines of code**, including comments, layers, weight initialization, and forward propagation with training & inference compatibility;
   - all done without using Hugging Face `transformers`. 🤯

6. **Infinite batch size**
   - regular batch size (limited) * gradient accumulation steps (unlimited) = effective batch size (unlimited)



## SoTA Features

- **[Oct 17, 2024]**  
  🚀 Implemented [Learn, Focus, and Review](https://arxiv.org/pdf/2409.06131) in Quokka.  
  🏆 20x (40x with SophiaG) speedup compared to OpenAI's GPT2 pretraining,  
  📈 Requiring only 5% of forward pass to achieve lower (better) perplexity scores across 4 datasets.

- **[Oct 15, 2024]**  
  🚀 Implemented GPT-4o, GPT-4/GPT-3.5, and GPT-3 tokenization using [tiktoken](https://github.com/openai/tiktoken).  
  🏆 3-6x faster than Hugging Face tokenizers.  
  ✌️ No license restrictions like Llama or Qwen.

- **[Oct 15, 2024]**  
  🚀 Implemented [Grouped Query Attention](https://arxiv.org/pdf/2305.13245) in Quokka.

- ~~**[Oct 14, 2024]**  
  🔍 Explored [Zigzag Ring Attention with Flash Attention](https://github.com/zhuzilin/ring-flash-attention).  
  ❗️No improvements on single-GPU training, increased VRAM usage by 0.4 GB for 2048 context length. Only meant for distributed GPUs.~~

- **[Oct 12, 2024]**  
  🎯 Achieved a perplexity score of **1.0** 🤯 on 80/20 train/val splits with a subset of 30,000 samples from FineWeb.  
  🧠 Pretrained with Multiquery Attention, 32 effective batch size, 2 epochs, 350M parameters, 2048 context length, and 12.6 GB VRAM. Works on free Google Colab!

- **[Oct 12, 2024]**  
  🏆 Applied [Sophia](https://arxiv.org/pdf/2305.14342), achieving 2x speedup and less total compute compared to Adam.  
  🔄 Applied token embedding rotation technique **RoPE** from [Llama 2 by Meta AI](https://arxiv.org/pdf/2307.09288).  
  📈 Reduced complexity and improved performance, especially for longer sequences.


## Walkthrough

1. (optional) Installing Flash Attention on Windows

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install ninja==1.11.1
pip install flash-attn --no-build-isolation
```

Compilation might take a while (~2 hours) on Windows, depending on your system specs.

Please read [this issue](https://github.com/Dao-AILab/flash-attention/issues/553) for other setups that worked on Windows.

2. Build

```bash
pip install git+https://github.com/aistrova/quokka.git
```

3. Train

```python
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
    val_dataloader=val_dataloader,
    tokenizer=tokenizer
)

# start training
trainer.run(MODEL_NAME)
```

4. Generate text

```python
import torch
from quokka.model import Quokka
from quokka.text_generation import AutoregressiveWrapper 
from quokka.data import get_tokenizer
from quokka.config import Quokka230MConfig, PretrainConfig

tokenizer = get_tokenizer("p50k_base")

config = Quokka230MConfig(tokenizer.n_vocab, PretrainConfig.max_ctx_len)

model = Quokka(config, pretrained=True)
checkpoint = torch.load('Quokka_epoch8.pth', weights_only=False)
model.load_state_dict(checkpoint['model'])
model.eval()

pad_value = tokenizer.max_token_value+1
wrapper = AutoregressiveWrapper(model, pad_value=pad_value)

# encode the prompt with bos
prompt_text = "let him cook"
prompt_tokens = torch.tensor([tokenizer.bos] + tokenizer.encode(prompt_text))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
prompt_tokens = prompt_tokens.to(device)

generated_tokens = wrapper.generate(
    prompts=prompt_tokens,
    temperature=0.7,
    seq_len=50,
    eos_token=tokenizer.eos
).tolist()
generated_tokens = [token for token in generated_tokens if token != pad_value and token not in tokenizer.special_tokens_set]
generated_text = tokenizer.decode(generated_tokens)  # , skip_special_tokens=True

print("Generated Text:")
print(generated_text)
```

## SoTA models

GPT3 with RoPE = GPT3 Plus
```python
from quokka.config import (
    GPT3Plus_125M_Config,
    GPT3Plus_350M_Config,
    GPT3Plus_760M_Config,
    GPT3Plus_1B_Config,
    GPT3Plus_3B_Config,
    GPT3Plus_6B_Config,
    GPT3Plus_13B_Config,
    GPT3Plus_175B_Config
)
```

## TODO

[x] - Import GPT3 Plus - GPT3 with RoPE  
[ ] - Import Llama 3.1, Qwen 2.5 as components  
[ ] - Reduce pretraining memory by half  
[ ] - Finetuning  
[ ] - Multimodal  
[ ] - Documentation  

<br>

## License

Quokka, by AIstrova, is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).  

Please attribute if you can to advocate for transparency, ethics, and integrity in AI development and technologies.

An attribution as simple as `Built with [Quokka](https://github.com/aistrova/quokka)` or `Inspired by [Quokka](https://github.com/aistrova/quokka)` suffice.  


## Honorable mentions

Thanks to [Fred](https://www.linkedin.com/in/fredzhang7) for leading this initiative, innovating all the Educational Features, and implementing all the SoTA Features up to 10/18/2024.


<br>

--

Ex-Google CEO said that AI will make the rich richer and the poor poorer.

At AIstrova, we do not wish for this future. Let's infer and reproduce top closed-source AI (across 6 modalities) to run fast on consumer CPUs and GPUs together!