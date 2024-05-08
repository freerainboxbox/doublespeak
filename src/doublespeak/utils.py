from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import lzma
from nacl import pwhash, secret
from nacl import utils as nacl_utils

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


# NOTE: SecretMessage is meant to be immutable
class SecretMessage:
    ciphertext: str
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
        sm.ciphertext = salt + ciphertext
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

class HiddenMonologue:
    def __init__(self, tokenizer: AutoTokenizer, model, system_prompt: str, password: str):
        self.tokenizer = tokenizer
        self.model = model
        # Set chat template with system prompt, and empty first message for assistant to go first
        self.chat = tokenizer.tokenize("<|system|>"+system_prompt+"<|end|>\n<|assistant|>")
        self.password = password
    def encode(self, plaintext):
        sm = SecretMessageFactory.fromPlaintext(plaintext, self.password)
        ciphertext = struct.unpack("?", sm.ciphertext)
        cur_index = 0
        # Create a fill_mask pipeline
        fill_mask = pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer, top_k=30, device=-1)
        while cur_index < len(ciphertext):
            # Get the next 4 bits
            bits = ciphertext[cur_index:cur_index+4]
            # Get the indices of the codebook that match the bits (checks up to size of code)
            indices = [i for i, x in enumerate(codebook) if x == bits]
            # Get the tokens that match the indices
            tokens = [self.tokenizer.decode(i) for i in indices]
            # Zero out all other logits
            for i in range(len(fill_mask.tokenizer)):
                if i not in indices:
                    fill_mask.logits[i] = 0
            # Renormalize the logits to sum to 1.0
            fill_mask.logits = fill_mask.logits / sum(fill_mask.logits)
            # Sample from the distribution
            token = fill_mask()
            # Append the token to the chat
            self.chat.append(token)
            cur_index += 4
            #TODO Validate this code



    def decode(self, ciphertext):
        pass
        # TODO finish decode method
    


class HiddenConversation:
    def __init__(self, tokenizer, model, system_prompt: str, password: str):
        self.tokenizer = tokenizer
        self.model = model
        # TODO finish constructor