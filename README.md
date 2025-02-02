# Spell Correction with Seq2Seq and Attention (Character-Level)

This repository provides a PyTorch implementation of a character-level sequence-to-sequence (Seq2Seq) model with an attention mechanism for correcting spelling errors in words and texts. The model is particularly useful for tasks such as fixing spell errors where special characters might be replaced by their simpler counterparts (e.g., `ş` → `s`, `ç` → `c`, etc.).

---

## Features

- **Seq2Seq Architecture:**  
  Combines an LSTM-based encoder and decoder to transform an input sequence (a word with errors) into a corrected output sequence.

- **Character-Level Processing:**  
  Works at the individual character level, making the model robust to out-of-vocabulary words and enabling fine-grained correction.

- **Attention Mechanism:**  
  Implements Bahdanau-style attention that allows the decoder to focus on relevant parts of the input during generation, leading to improved accuracy.


---

## Model Architecture

### Encoder
- **Type:** LSTM-based encoder
- **Function:** Transforms a sequence of character embeddings into hidden states.

### Decoder
- **Type:** LSTM-based decoder
- **Function:** Uses the encoder’s hidden states along with a Bahdanau-style attention mechanism to generate the corrected word one character at a time.

### Attention
- **Mechanism:** Bahdanau-style attention
- **Function:** Computes a context vector for each decoding step by weighing the encoder outputs. This helps the decoder focus on the most relevant parts of the input sequence.

## Text Preprocessing

### `text_to_tensor`
- **Purpose:** Converts an input string into a tensor of indices.
- **Additional Tasks:**
  - Adds special tokens for the beginning (`<bos>`) and end (`<eos>`) of the sequence.
  - Pads the sequence to a fixed length.

### `tensor_to_text`
- **Purpose:** Converts a tensor of indices back into a string.
- **Behavior:** Stops conversion when encountering `<eos>` or `<pad>` tokens.

## Correction Functions

### `correct_word`
- **Purpose:** Processes a single word.
- **Process:**
  1. Converts the word into a tensor.
  2. Runs it through the encoder and decoder.
  3. Converts the output tensor back into a corrected word.

### `correct_text`
- **Purpose:** Processes full text.
- **Process:**
  1. Splits the full text into tokens (words and delimiters).
  2. Converts words to lower case.
  3. Corrects each word individually using `correct_word`.
  4. Reconstructs the text while preserving its original structure (spaces, punctuation, etc.).


