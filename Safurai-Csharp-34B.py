# pip install transformers accelerate

from transformers import AutoTokenizer
import transformers
import torch

model = "Safurai/Evol-csharp-full"
prompt = "User: \n {how to publish a message in rabbitmq?} \n Assistant: "

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    f'{prompt}',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=1024,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
