from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct-onnx", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct-onnx", trust_remote_code=True)

class HiddenMonologue:
    def __init__(self, tokenizer, model, system_prompt: str, password: str):
        self.tokenizer = tokenizer
        self.model = model
        # Set chat template with system prompt, and empty first message for assistant to go first
        self.task = tokenizer(system_prompt, return_tensors="pt").to(model.device)
        self.password = password
    def encode(self, plaintext):
        pass
        # TODO finish encode method
    def decode(self, ciphertext):
        pass
        # TODO finish decode method
    


class HiddenConversation:
    def __init__(self, tokenizer, model, system_prompt: str, password: str):
        self.tokenizer = tokenizer
        self.model = model
        # TODO finish constructor