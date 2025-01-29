from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "mistralai/Mistral-7B-Instruct"  # 使用するLLM

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return model, tokenizer

def generate_text(model, prompt):
    model, tokenizer = model
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids)
    return tokenizer.decode(output[0], skip_special_tokens=True)
