from typing import List, Tuple, Optional
from secrets import randbelow
import lzma

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from nacl import pwhash, secret
from nacl import utils as nacl_utils
from nacl.exceptions import CryptoError
import blake3
import torch

CPU = -1
BLAKE3_DIGEST_SIZE = 32
BLAKE3_KEY_SIZE = 32

def _sec_randfloat() -> float:
    """Securely returns a uniformly random float in [0.0, 1.0), for selecting tokens from probability distribution."""
    return randbelow(2**32) / 2**32

# Read at current position of ciphertext the next 1, 2, 3, 4 bit windows, and the indices of the codebook that match tell us all eligible tokens.
# Then, zero out all other logits, and renormalize the logits to sum to 1.0. Then, sample from the distribution.
def gen_codebook(num_bits: int):
    codebook = []
    for bits in range(1, num_bits+1):
        for i in range(2**bits):
            code = tuple(bool(int(bit)) for bit in format(i, f"0{bits}b"))
            codebook.append(code)
    return tuple(codebook)

codebook = gen_codebook(4)

def pp_codebook_id(codebook_id: int) -> str:
    """Pretty-prints a codebook ID as a binary string."""
    output = "0b"
    for bit in codebook[codebook_id]:
        output += "1" if bit else "0"
    return output

CODEBOOK_SIZE = len(codebook)

def _bytes_to_bools(bytestream: bytes) -> List[bool]:
    """Converts a bytestream into a list of bits."""
    return [int(bit) for byte in bytestream for bit in f"{byte:08b}"]

def _bools_to_bytes(bits: List[bool]) -> bytes:
    """Converts a list of bits into a bytestream."""
    return bytes(int("".join(map(str, bits[i:i+8])), 2) for i in range(0, len(bits), 8))

# NOTE: _SecretMessage is meant to be immutable
class _SecretMessage:
    ciphertext: bytes
    plaintext: str

class _SecretMessageFactory:
    @staticmethod
    def from_plaintext(plaintext: str, password: str) -> _SecretMessage:
        """Encrypts a plaintext message using a password and returns a _SecretMessage object."""
        plaintext_bytes: bytes = plaintext.encode()
        # preset must be 9 as this process is already very space-inefficient
        plaintext_bytes: bytes = lzma.compress(plaintext_bytes, preset=9)
        kdf: function = pwhash.argon2id.kdf
        # fixed length, goes before ciphertext but is treated as portion of ciphertext to user
        salt: bytes = nacl_utils.random(pwhash.argon2id.SALTBYTES)
        key_symmetric: bytes = kdf(secret.SecretBox.KEY_SIZE, password.encode(), salt, pwhash.argon2id.OPSLIMIT_SENSITIVE, pwhash.argon2id.MEMLIMIT_SENSITIVE)
        key_mac: bytes = kdf(BLAKE3_KEY_SIZE, password.encode(), salt, pwhash.argon2id.OPSLIMIT_SENSITIVE, pwhash.argon2id.MEMLIMIT_SENSITIVE)
        box: secret.SecretBox = secret.SecretBox(key_symmetric)
        nonce: bytes = nacl_utils.random(secret.SecretBox.NONCE_SIZE)
        ciphertext: bytes = box.encrypt(plaintext_bytes, nonce)
        mac_chksum: bytes = blake3.blake3(ciphertext, key=key_mac).digest(length=BLAKE3_DIGEST_SIZE)
        sm = _SecretMessage()
        # Whenever 0x03 is found, the decoder will check if the following MAC is valid, terminating if it is.
        # This prevents the environment from figuring out that there is a hidden message since the MAC will appear random.
        sm.ciphertext = salt + ciphertext + bytes.fromhex("03") + mac_chksum
        sm.plaintext = plaintext
        return sm
    @staticmethod
    def from_ciphertext(ciphertext: bytes, password: str) -> _SecretMessage:
        """Takes a ciphertext bytestream directly obtained from decoding bytes from a token sequence and returns a _SecretMessage object."""
        salt = ciphertext[:pwhash.argon2id.SALTBYTES]
        ciphertext = ciphertext[pwhash.argon2id.SALTBYTES:]
        kdf = pwhash.argon2id.kdf
        key_symmetric = kdf(secret.SecretBox.KEY_SIZE, password.encode(), salt, pwhash.argon2id.OPSLIMIT_SENSITIVE, pwhash.argon2id.MEMLIMIT_SENSITIVE)
        key_mac = kdf(BLAKE3_KEY_SIZE, password.encode(), salt, pwhash.argon2id.OPSLIMIT_SENSITIVE, pwhash.argon2id.MEMLIMIT_SENSITIVE)
        box = secret.SecretBox(key_symmetric)
        eot_detected = False
        for i, read_head in enumerate(ciphertext[:-BLAKE3_DIGEST_SIZE-1]):
            if read_head == 0x03:
                eot_detected = True
                candidate_mac = ciphertext[i+1:i+1+BLAKE3_DIGEST_SIZE]
                computed_mac = blake3.blake3(ciphertext[:i], key=key_mac).digest(length=BLAKE3_DIGEST_SIZE)
                if candidate_mac == computed_mac:
                    sm = _SecretMessage()
                    sm.ciphertext = ciphertext[:i]
                    try:
                        sm.plaintext = lzma.decompress(box.decrypt(sm.ciphertext)).decode()
                        return sm
                    except CryptoError:
                        pass
        if not eot_detected:
            raise ValueError("No end-of-transmission marker found in ciphertext. Is the message corrupted?")
        raise ValueError("No valid MAC found in ciphertext. Is the message corrupted or was the password incorrect?")

def get_codebook_indices(ciphertext: List[bool], cur_index: int) -> List[int]:
    """Given a ciphertext and the read head position, returns the indices in the codebook that match the next 1, 2, 3, or 4 bits."""
    codebook_indices = []
    max_bits = min(4, len(ciphertext) - cur_index)
    for i in range(2**(max_bits+1)-2):
        valid = True
        for pos, bit in enumerate(codebook[i]):
            if bit != ciphertext[cur_index + pos]:
                valid = False
                break
        if valid:
            codebook_indices.append(i)
    if len(codebook_indices) > 4:
        return codebook_indices[:4] # BUG: For some reason, occasionally the codebook will contain a fifth [True, True, True, True] entry, but is otherwise correct.
    return codebook_indices
        

class HiddenMonologue:
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, system_prompt: str, password: str):
        self.tokenizer: AutoTokenizer = tokenizer
        self.model: AutoModelForCausalLM = model
        self.system_prompt: str = system_prompt
        # Set chat template with system prompt, and empty first message for assistant to go first
        self.chat: Optional[str] = None
        self.password = password
    def decode_recent_token(self, tokenized_chat, token_decoding_index: int) -> int:
        """Returns the index in codebook corresponding the last token of the tokenized running chat."""
        tokens_so_far = tokenized_chat
        tokens_so_far["input_ids"] = tokens_so_far["input_ids"][:token_decoding_index]
        with torch.no_grad():
            logits = self.model(**tokens_so_far).logits
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_idxs = torch.topk(probs[0, -1], k=CODEBOOK_SIZE)
        # Sort in descending order of probabilities
        top_probs, top_idxs = zip(*sorted(zip(top_probs, top_idxs), reverse=True))
        # Find the index of the token in the codebook
        for i in range(CODEBOOK_SIZE):
            if codebook[top_idxs[i]] == tokens_so_far["input_ids"][-1]:
                return i
    def encode_bits(self, ciphertext: List[bool], cur_index: int, context: str) -> Tuple[str, int]:
        """At the current position of the ciphertext, return a decoded token and the number of bits consumed probabalistically."""
        context_tokens = self.tokenizer(context, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**context_tokens).logits
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_idxs = torch.topk(probs[0, -1], k=CODEBOOK_SIZE)
        # Sort in descending order of probabilities
        top_probs, top_idxs = zip(*sorted(zip(top_probs, top_idxs), reverse=True))
        top_probs = list(top_probs)
        top_idxs = list(top_idxs)
        # Determine possible codebook indices
        codebook_indices = get_codebook_indices(ciphertext, cur_index)
        # Zero out all scores except for the ones that match the codebook
        total = 0
        for i in range(CODEBOOK_SIZE):
            if i not in codebook_indices:
                top_probs[i] = 0
            else:
                total += top_probs[i]
        # Renormalize the scores to sum to 1.0
        for i in range(CODEBOOK_SIZE-1):
            top_probs[i] /= total
        # Ensure sum is 1.0 in case of floating point errors
        top_probs[-1] = 1 - sum(top_probs[:-1])
        # Show eligible tokens
        print("[")
        for i in range(CODEBOOK_SIZE):
            if top_probs[i] > 0:
                print(f"    {i} (codeword: {pp_codebook_id(i)}, {top_probs[i]*100:.2f}%): \"{self.tokenizer.decode(top_idxs[i], clean_up_tokenization_spaces=False)}\",")
        print("],")
        # Randomly sample from the distribution
        rand = _sec_randfloat()
        for i in range(CODEBOOK_SIZE):
            if top_probs[i] >= rand:
                print(f"Selected token: {i} (codeword: {pp_codebook_id(i)}, {top_probs[i]*100:.2f}%): \"{self.tokenizer.decode(top_idxs[i], clean_up_tokenization_spaces=False)}\"\n-----------------")
                return self.tokenizer.decode(top_idxs[i], clean_up_tokenization_spaces=False), len(codebook[i])
            rand -= top_probs[i]
    def reset_chat(self):
        self.chat = "<|system|>"+self.system_prompt+"<|end|>\n<|assistant|>"
    def encode(self, plaintext) -> str:
        """Encrypts a plaintext message into ciphertext using the password, then encodes it into a monologue."""
        # TODO: It turns out that using text directly is lossy and doesn't preserve spaces. This has to be refactored to use token lists and only decode at the final step. Add a token list to the class and refactor the code to use it.
        self.reset_chat()
        sm = _SecretMessageFactory.from_plaintext(plaintext, self.password)
        ciphertext = _bytes_to_bools(sm.ciphertext)
        cur_index = 0
        while cur_index < len(ciphertext):
            token, consumed_bits = self.encode_bits(ciphertext, cur_index, self.chat)
            self.chat += token
            cur_index += consumed_bits
        # If we are not at <|end|> yet, continue the conversation with normal generation
        if "<|end|>" not in self.chat:
            self.chat += self.tokenizer.decode(self.model.generate(self.tokenizer(self.chat, return_tensors="pt")["input_ids"], max_length=512, do_sample=True, pad_token_id=self.tokenizer.eos_token_id, num_return_sequences=1)[0], clean_up_tokenization_spaces=False)
        return self.chat
    def decode(self, llm_generated_text: str) -> str:
        """Decodes a hidden plaintext message from a block of text, encrypted."""
        self.reset_chat()
        tokenized_chat = self.tokenizer(self.chat, return_tensors="pt")
        cur_index = 0
        ciphertext: List[bool] = []
        # Decode recent tokens until we reach the end of the tokenized chat. At each iteration, add the corresponding codebook entry to the ciphertext.
        for token_decoding_index in range(tokenized_chat["input_ids"]):
            ciphertext += codebook[self.decode_recent_token(tokenized_chat, token_decoding_index)]
        # Pack into a bytestring, and then decrypt
        sm = _SecretMessageFactory.from_ciphertext(_bools_to_bytes(ciphertext), self.password)
        return sm.plaintext

class HiddenConversation:
    def __init__(self, tokenizer, model, system_prompt: str, password: str):
        self.tokenizer = tokenizer
        self.model = model
        # TODO finish constructor