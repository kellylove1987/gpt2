from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline

model_id = "deepseek-ai/DeepSeek-Prover-V1.5-RL"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
streamer = TextStreamer(tokenizer)
pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, streamer=streamer
)
text = """import Mathlib

theorem quadratic (x : Real) (hx : x * x + 2 * x + 1 = 0) : x = -1 := by"""
outputs = pipe(text, max_new_tokens=256)
response = outputs[0]["generated_text"]
print(response)
