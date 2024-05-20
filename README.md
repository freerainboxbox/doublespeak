# doublespeak
_work in progress_

Hide secrets in innocuous communication using the power of LLMs.

Inspired by [LLuffman](https://botnoise.org/~pokes/lluffman/)

## How does this work?

Using the lightweight [Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) model and a shared system prompt to complete, we do the following to hide text inside an innocuous block of LLM generated text:

1. Compress our text using `lzma`: `comp_text = lzma(text)`
2. Encrypt the text using `(List[bool]) ciphertext = AES_E(comp_text, secret)`
3. Format message as `salt || ciphertext || 0x03 || MAC(ciphertext, secret)`
4. Generate the top-30 tokens with associated probabilities and apply a fixed code table: `{0b0: token[1], 0b1: token[2], ... 0b1111: token}`
5. Zero out ineligible tokens, renormalize
6. Sample the next token with modified probabilities and increment ciphertext pointer accordingly
7. Repeat 4-6 until message exhausted
8. Continue generating top-30 tokens with randomness until `<|endoftext|>`
9. Return the final encoded block of text, and send over an insecure channel.

We can then decode by reverse-matching the predictions to our code-table to reconstruct the ciphertext, stopping at `0x03`, then decrypting and uncompressing. If the MAC doesn't match, keep trying every `0x03` until it works, otherwise return an error.

Conversations may also work, as long as total conversation length + system prompt does not exceed the 128k window. The ciphertext is always aligned at the message boundary in this case.

It may also be possible to do an ad-hoc key exchange for parties that cannot securely establish a shared secret beforehand. Some public form of randomness that is slow to change would work as a "key" in this case, used to share public parameters. The easiest method of doing this is simply using the [most recent Bitcoin block hash](https://blockchair.com/bitcoin/blocks) as a 256-bit key.