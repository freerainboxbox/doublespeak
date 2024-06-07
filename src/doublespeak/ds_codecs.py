from dataclasses import dataclass
from typing import List, Tuple, Optional
from secrets import randbelow
import lzma

from transformers import AutoTokenizer, AutoModelForCausalLM
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

def _gen_codebook(num_bits: int):
    cb = []
    for bits in range(1, num_bits+1):
        for i in range(2**bits):
            code = tuple(bool(int(bit)) for bit in format(i, f"0{bits}b"))
            cb.append(code)
    return tuple(cb)

codebook = _gen_codebook(4)

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
@dataclass
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
        kdf: callable = pwhash.argon2id.kdf
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
        return codebook_indices[:3] # BUG: For some reason, occasionally the codebook will contain a fifth [True, True, True, True] entry, but is otherwise correct.
    return codebook_indices
        

class HiddenMonologue:
    """Represents a secure abstraction for hiding a secret in a single instance of a large language model's output.
    Can be used to send a single message notoriously over an insecure channel, or for hiding seemingly innocuous data at rest."""
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, system_prompt: str, password: str, temperature: float = 5.0):
        """
        Initializes a HiddenMonologue object with a tokenizer, model, system prompt, and password.
        Inputs:
        - tokenizer: The tokenizer to use for encoding and decoding text (see utils.py for example).
        - model: The model to use for encoding and decoding text (see utils.py for example). Tokenizer should be derived from this model.
        - system_prompt: The system prompt to use for the chat. Good system prompts nudge the assistant to generate natural text with little to no special formatting, and specifically prompts the assistant to generate one thing only.
            - Good Example: "Assistant writes a simple travel guide for Venice."
            - Bad Example: "Typeset the heat equation in LaTeX."
        - password: The shared secret password to use for authenticated encryption and decryption of hidden messages.
        - temperature (default 5.0): Creativity of the output text.
            - Higher values increase encoding efficiency by giving higher probability to longer codewords (which consume more bits), while lower values may produce more logical text.
            - Values too low may cause strange sounding text since rejection sampling is used, giving a mix of coherent text and gibberish.
        
        Note that identical objects with identical plaintext will not produce identical outputs, as multiple computations are probabilistic.
        """
        # USER: For custom temperatures, it must be a shared secret, as top-k ordering is not invariant to temperature
        self.temperature = temperature
        self.coldness = 1.0 / temperature # For logits multiply (faster than division)
        self.tokenizer: AutoTokenizer = tokenizer
        self.model: AutoModelForCausalLM = model
        self.system_prompt: str = system_prompt
        # Set chat template with system prompt, and empty first message for assistant to go first
        self.chat: Optional[str] = None
        self.token_sequence: Optional[List[int]] = None
        self.output_start_index: Optional[int] = None # Index of the first token of the assistant's output (after <|assistant|>)
        self.password = password
    def decode_recent_token(self, tokenized_chat, token_decoding_index: int) -> int:
        """Returns the index in codebook corresponding the last token of the tokenized running chat."""
        tokens_so_far = tokenized_chat
        tokens_so_far["input_ids"] = tokens_so_far["input_ids"][:token_decoding_index]
        with torch.no_grad():
            logits = self.model(**tokens_so_far).logits
            # Apply temperature scaling (logits * coldness)
            torch.mul(logits, self.coldness, out=logits)
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_idxs = torch.topk(probs[0, -1], k=CODEBOOK_SIZE)
        # Sort in descending order of probabilities
        top_probs, top_idxs = zip(*sorted(zip(top_probs, top_idxs), reverse=True))
        # Find the index of the token in the codebook
        for i in range(CODEBOOK_SIZE):
            if codebook[top_idxs[i]] == tokens_so_far["input_ids"][-1]:
                return i
    def encode_bits(self, ciphertext: List[bool], cur_index: int, context_tokens: List[str]) -> Tuple[str, int]:
        """
        At the current position of the ciphertext, return a decoded token and the number of bits consumed probabalistically.
        Inputs:
        - ciphertext: A list of bools, containing the raw encoding of the text (bits grow from LSB to MSB).
        - cur_index: The current read head position in the ciphertext.
        - context_tokens: Pre-tokenized context, containing up to all bits already encoded.

        Outputs:
        - token: The text (decoded) token to append to running state for iteration
        - consumed_bits: The number of bits consumed by the token
        """
        # Prepare context for inference (requires a list of strings to satisfy API)
        context: dict = self.tokenizer(context_tokens, return_tensors="pt", is_split_into_words=True)
        with torch.no_grad():
            logits = self.model(**context).logits
            # Apply temperature scaling (logits * coldness)
            torch.mul(logits, self.coldness, out=logits)
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_idxs = torch.topk(probs[0, -1], k=CODEBOOK_SIZE)
        # Sort in descending order of probabilities
        top_probs, top_idxs = zip(*sorted(zip(top_probs, top_idxs), reverse=True))
        top_probs = list(top_probs)
        top_idxs = list(top_idxs)
        top_probs = [tensor.item() for tensor in top_probs]
        top_idxs = [tensor.item() for tensor in top_idxs]
        # Determine possible codebook indices
        codebook_indices = get_codebook_indices(ciphertext, cur_index)
        # Zero out all scores except for the ones that match the codebook
        total = 0
        for i in range(CODEBOOK_SIZE):
            # Forbid EOS token to continue generation
            if i not in codebook_indices or top_idxs[i] == self.tokenizer.eos_token_id:
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
                print(f"Selected token: {i} (codeword: {pp_codebook_id(i)}, {top_probs[i]*100:.2f}%): \"{self.tokenizer.decode(top_idxs[i], clean_up_tokenization_spaces=False)}\"\n{len(ciphertext)-cur_index-len(codebook[i])} bits remain\n-----------------")
                return self.tokenizer.convert_ids_to_tokens(top_idxs[i]), len(codebook[i])
            rand -= top_probs[i]
    def reset_chat(self):
        """Resets the chat to the system prompt."""
        self.chat = "<|system|>"+self.system_prompt+"<|end|>\n<|assistant|>"
        self.token_sequence = self.tokenizer(self.chat, return_tensors="pt")["input_ids"].tolist()[0]
        self.output_start_index = len(self.token_sequence)
    def encode(self, plaintext: str) -> str:
        """
        Encrypts a plaintext message into ciphertext using the password, then encodes it into a monologue.
        Chat state and internal token sequence is mutated to contain formatted system prompt concatenated with generated text.

        Inputs:
        - plaintext: The plaintext message to encode
        Outputs:
        - The encoded monologue containing the ciphertext, without formatting
        """
        self.reset_chat()
        # Our running state, which starts with an empty chat containing the system prompt
        context_tokens = self.tokenizer.convert_ids_to_tokens(self.token_sequence)
        sm = _SecretMessageFactory.from_plaintext(plaintext, self.password)
        ciphertext = _bytes_to_bools(sm.ciphertext)
        cur_index = 0
        while cur_index < len(ciphertext):
            token, consumed_bits = self.encode_bits(ciphertext, cur_index, context_tokens)
            context_tokens.append(token)
            self.token_sequence.append(self.tokenizer.convert_tokens_to_ids(token))
            cur_index += consumed_bits
        # Continue generation (we are guaranteed to not have EOS yet)
        inputs = self.tokenizer(context_tokens, return_tensors="pt", is_split_into_words=True)
        output = self.model.generate(**inputs, max_length=512, do_sample=True, pad_token_id=self.tokenizer.eos_token_id, num_return_sequences=1)[0]
        self.token_sequence = output.tolist()
        self.chat = self.tokenizer.decode(self.token_sequence, clean_up_tokenization_spaces=False)
        return self.tokenizer.decode(self.token_sequence[:self.output_start_index], clean_up_tokenization_spaces=False)
    def decode(self, llm_generated_text: str) -> str:
        """Decodes a hidden plaintext message from a block of text, encrypted."""
        self.reset_chat()
        tokenized_chat = self.tokenizer(self.chat, return_tensors="pt")
        tokens_to_add = self.tokenizer(llm_generated_text, return_tensors="pt")["input_ids"]
        ciphertext: List[bool] = []
        # Decode recent tokens until we reach the end of the tokenized chat. At each iteration, add the corresponding codebook entry to the ciphertext.
        for i, token_to_add in enumerate(tokens_to_add):
            token_decoding_index = len(tokenized_chat["input_ids"]) + i
            token = self.decode_recent_token(tokenized_chat, token_decoding_index)
            ciphertext += codebook[token]
            tokenized_chat["input_ids"] = torch.cat((tokenized_chat["input_ids"], token_to_add.unsqueeze(0)), dim=1)
        # Pack into a bytestring, and then decrypt
        sm = _SecretMessageFactory.from_ciphertext(_bools_to_bytes(ciphertext), self.password)
        return sm.plaintext[self.output_start_index:]

class HiddenConversation(HiddenMonologue):
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, system_prompt: str, password: str):
        super().__init__(tokenizer, model, system_prompt, password)
    def encode_message(self, plaintext: str) -> str:
        """Generate message from secret plaintext and append to chat. Return encoded new message."""
        pass
    def push_message(self, message: str) -> str:
        """Push known secret message from other party into chat. Return secret plaintext message."""
        pass