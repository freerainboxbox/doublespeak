# doublespeak
_work in progress_

Hide secrets in innocuous communication using the power of LLMs.

Inspired by [LLuffman](https://botnoise.org/~pokes/lluffman/)

## How does this work?

Using the lightweight [Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) model and a shared system prompt to complete, we do the following to hide text inside an innocuous block of LLM generated text:

1. Compress our text using `lzma`: `comp_text = lzma(text)`
2. Demarcate text boundary using `comp_text = comp_text||0x03`
3. Encrypt the text using `(List[bool]) ciphertext = AES_E(comp_text, secret)`
4. Generate the top-30 tokens with associated logits and apply a fixed code table: `{0b0: token[1], 0b1: token[2], ... 0b1111: token}`
5. Filter out all codes that do not match the current read head of the ciphertext by applying -inf to ineligible logits (including premature ending of text).
6. Compute the next token with modified logits and increment ciphertext pointer accordingly
7. Repeat 4-6 until ciphertext exhausted
8. Continue generating top-30 tokens with randomness until `<|endoftext|>`

We can then decode by reverse-matching the predictions to our code-table to reconstruct the ciphertext, stopping at `0x03`, then decrypting and uncompressing.

Conversations may also work, as long as total conversation length + system prompt does not exceed the 128k window. The ciphertext is always aligned at the message boundary in this case.