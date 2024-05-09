from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import lzma
from nacl import pwhash, secret
from nacl import utils as nacl_utils
from typing import List
from secrets import randbelow
CPU = -1

def sec_randfloat() -> float:
    return randbelow(2**32) / 2**32

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)

# Read at current position of ciphertext the next 1, 2, 3, 4 bit windows, and the indices of the codebook that match tell us all eligible tokens.
# Then, zero out all other logits, and renormalize the logits to sum to 1.0. Then, sample from the distribution.
codebook = (
    (False),
    (True),
    (False, False),
    (False, True),
    (True, False),
    (True, True),
    (False, False, False),
    (False, False, True),
    (False, True, False),
    (False, True, True),
    (True, False, False),
    (True, False, True),
    (True, True, False),
    (True, True, True),
    (False, False, False, False),
    (False, False, False, True),
    (False, False, True, False),
    (False, False, True, True),
    (False, True, False, False),
    (False, True, False, True),
    (False, True, True, False),
    (False, True, True, True),
    (True, False, False, False),
    (True, False, False, True),
    (True, False, True, False),
    (True, False, True, True),
    (True, True, False, False),
    (True, True, False, True),
    (True, True, True, False),
    (True, True, True, True),
)

CODEBOOK_SIZE = len(codebook)

# NOTE: SecretMessage is meant to be immutable
class SecretMessage:
    ciphertext: bytes
    plaintext: str

class SecretMessageFactory:
    @staticmethod
    def fromPlaintext(plaintext: str, password: str) -> SecretMessage:
        plaintext_bytes = plaintext.encode()
        # preset must be 9 as this process is already very space-inefficient
        plaintext_bytes = lzma.compress(plaintext_bytes, preset=9)
        kdf = pwhash.argon2id.kdf
        # fixed length, goes before ciphertext but is treated as portion of ciphertext to user
        salt = nacl_utils.random(pwhash.argon2id.SALTBYTES)
        key = kdf(secret.SecretBox.KEY_SIZE, password.encode(), salt, pwhash.argon2id.OPSLIMIT_SENSITIVE, pwhash.argon2id.MEMLIMIT_SENSITIVE)
        box = secret.SecretBox(key)
        nonce = nacl_utils.random(secret.SecretBox.NONCE_SIZE)
        ciphertext = box.encrypt(plaintext_bytes, nonce)
        sm = SecretMessage()
        sm.ciphertext = salt + ciphertext + bytes.fromhex("03")
        sm.plaintext = plaintext
        return sm
    @staticmethod
    def fromCiphertext(ciphertext: str, password: str) -> SecretMessage:
        salt = ciphertext[:pwhash.argon2id.SALTBYTES]
        ciphertext = ciphertext[pwhash.argon2id.SALTBYTES:]
        kdf = pwhash.argon2id.kdf
        key = kdf(secret.SecretBox.KEY_SIZE, password.encode(), salt, pwhash.argon2id.OPSLIMIT_SENSITIVE, pwhash.argon2id.MEMLIMIT_SENSITIVE)
        box = secret.SecretBox(key)
        plaintext_bytes = box.decrypt(ciphertext)
        plaintext_bytes = lzma.decompress(plaintext_bytes)
        plaintext = plaintext_bytes.decode()
        sm = SecretMessage()
        sm.ciphertext = ciphertext
        sm.plaintext = plaintext
        return sm

def get_codebook_indices(ciphertext: List[bool], cur_index: int) -> list:
    codebook_indices = None
    max_bits = min(4, len(sm.ciphertext) - cur_index)
    match max_bits:
        case 1:
            codebook_indices = ciphertext[cur_index]
        case 2:
            bits = ciphertext[cur_index:cur_index+2]
            for i, code in enumerate(codebook[:5]):
                if code == bits:
                    codebook_indices.append(i)
        case 3:
            bits = ciphertext[cur_index:cur_index+3]
            for i, code in enumerate(codebook[:13]):
                if code == bits:
                    codebook_indices.append(i+5)
        case 4:
            bits = ciphertext[cur_index:cur_index+4]
            for i, code in enumerate(codebook):
                if code == bits:
                    codebook_indices.append(i+13)
    return codebook_indices

class HiddenMonologue:
    def __init__(self, tokenizer: AutoTokenizer, model, system_prompt: str, password: str):
        self.tokenizer = tokenizer
        self.model = model
        self.system_prompt = system_prompt
        # Set chat template with system prompt, and empty first message for assistant to go first
        self.chat = None
        self.password = password
    def reset_chat(self):
        self.chat = "<|system|>"+self.system_prompt+"<|end|>\n<|assistant|>"
    def encode(self, plaintext) -> str:
        self.reset_chat()
        sm = SecretMessageFactory.fromPlaintext(plaintext, self.password)
        ciphertext = struct.unpack("?", sm.ciphertext)
        cur_index = 0
        # Create a fill_mask pipeline
        fill_mask = pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer, top_k=CODEBOOK_SIZE, device=CPU)
        while cur_index < len(ciphertext):
            next_tokens = fill_mask(self.chat+"<mask>")
            # Zero out all scores except for the ones that match the codebook
            codebook_indices = get_codebook_indices(ciphertext, cur_index)
            total = 0
            for i in range(CODEBOOK_SIZE):
                if i not in codebook_indices:
                    next_tokens[0]["scores"][i] = 0
                else:
                    total += next_tokens[0]["scores"][i]
            # Renormalize the scores to sum to 1.0
            for i in range(CODEBOOK_SIZE):
                next_tokens[0]["scores"][i] /= total
            # Randomly sample from the distribution
            rand = sec_randfloat()
            # Find smallest score above or equal to rand
            for i in range(CODEBOOK_SIZE):
                if next_tokens[0]["scores"][i] >= rand:
                    break
                rand -= next_tokens[0]["scores"][i]
            cur_index += len(codebook[codebook_indices[i]])
            self.chat = next_tokens[i]["sequence"]
        # Complete the chat normally using text-generation
        generate_padding = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, num_return_sequences=1, device=CPU)
        self.chat = generate_padding(self.chat)[0]["generated_text"]
        return self.chat

    def decode(self, llm_generated_text: str) -> str:
        self.reset_chat()
        self.chat += llm_generated_text
        # TODO: finish decode method



class HiddenConversation:
    def __init__(self, tokenizer, model, system_prompt: str, password: str):
        self.tokenizer = tokenizer
        self.model = model
        # TODO finish constructor