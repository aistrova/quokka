import torch
from quokka.model import Quokka
from quokka.text_generation import AutoregressiveWrapper 
from quokka.data import get_tokenizer
from quokka.config import Quokka230MConfig, PretrainConfig

tokenizer = get_tokenizer()

config = Quokka230MConfig(tokenizer.n_vocab, PretrainConfig.max_ctx_len)

model = Quokka(config, pretrained=True)
checkpoint = torch.load('Quokka_epoch8.pth', weights_only=False)
model.load_state_dict(checkpoint['model'])
model.eval()

pad_value = tokenizer.max_token_value+1
wrapper = AutoregressiveWrapper(model, pad_value=pad_value)

# encode the prompt with bos
prompt_text = "he is running for"
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
