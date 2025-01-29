from fastapi import FastAPI
from app.model import load_model, generate_text

app = FastAPI()

model = load_model()

@app.get("/")
def read_root():
    return {"message": "LLM Server is running"}

@app.post("/generate")
def generate(prompt: str):
    return {"response": generate_text(model, prompt)}
