import ds_codecs
from transformers import AutoTokenizer, AutoModelForCausalLM

# USER: You MUST specify a revision to ensure reproducibility
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True, revision="8a362e755d2faf8cec2bf98850ce2216023d178a")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True, revision="8a362e755d2faf8cec2bf98850ce2216023d178a")

def encode_one(plaintext: str, prompt: str, password: str) -> str:
    hidden_monologue = ds_codecs.HiddenMonologue(tokenizer, model, prompt, password)
    return hidden_monologue.encode(plaintext)

def decode_one(llm_generated_text: str, prompt: str, password: str) -> str:
    hidden_monologue = ds_codecs.HiddenMonologue(tokenizer, model, prompt, password)
    return hidden_monologue.decode(llm_generated_text)