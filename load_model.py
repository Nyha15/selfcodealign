from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "./stable-code-instruct-3b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype="auto"
)

print("Model loaded successfully")
