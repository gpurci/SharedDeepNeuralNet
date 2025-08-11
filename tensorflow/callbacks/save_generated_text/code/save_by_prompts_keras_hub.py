#!/usr/bin/python

from tensorflow.keras.callbacks import Callback
from pathlib import Path
import sys
import keras_hub

class SaveByPromptsKerasHub(Callback):
   """A callback to save generated text from a trained model.
   1. Feed some starting prompt to the model
   2. Predict probabilities for the next token
   3. Sample the next token and add it to the next input

   Arguments:
     max_tokens: Integer, the number of tokens to be generated after prompt.
     start_tokens: List of integers, the token indices for the starting prompt.
     vocab: List of strings, obtained from the TextVectorization layer.
     top_k: Integer, sample from the 'top_k' token predictions.
     print_every: Integer, print after this many epochs.
   """

   def __init__(self, path, tokenizer, sampler, prompts, seq_size, maxlen=10, print_every=1):
      self.path = Path(path)
      self.path.mkdir(mode=0o777, parents=True, exist_ok=True)
      self.tokenizer = tokenizer
      self.sampler   = sampler
      self.seq_size  = seq_size
      self.maxlen    = maxlen
      self.prompts   = prompts
      self.print_every = print_every

   def __prompt_tokenizer(self, prompts):
      print(prompts)
      self.prompt_tokens = []
      for prompt in prompts:
         # Tokenize the text
         tokens     = self.tokenizer.tokenize(prompt)
         # Convert tokens to their corresponding IDs in the vocabulary
         tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
         self.prompt_tokens.append(tokens_ids)

   def __input_tokenize(self, prompt):
      # Tokenize the text
      tokens_ids = self.tokenizer(
                       prompt,
                       add_special_tokens=True,
                       max_length=self.seq_size,  # Example max length
                       padding="max_length",
                       truncation=True,
                       return_tensors="tf"
                   )
      return tokens_ids["input_ids"]


   def _prompt_tokenizer(self, prompts):
      # Tokenize the text, adding special tokens and preparing for the model
      # `add_special_tokens=True` ensures [CLS] and [SEP] are added.
      # `max_length` truncates/pads to a fixed length (important for batches).
      # `padding='max_length'` adds [PAD] tokens.
      # `truncation=True` truncates if the text is too long.
      # `return_tensors='pt'` returns PyTorch tensors (you could use 'tf' for TensorFlow).
      self.prompt_tokens = []
      for prompt in prompts:
         # Tokenize the text
         tokens_ids = self.tokenizer(
                          prompt,
                          add_special_tokens=True,
                          max_length=self.seq_size,  # Example max length
                          padding="max_length",
                          truncation=True,
                          return_tensors="tf"
                      )
         self.prompt_tokens.append(tokens_ids["input_ids"])

   def on_epoch_end(self, epoch, logs=None):
      if (((epoch + 1) % self.print_every) == 0):
         path_gen = self.path.joinpath("name").with_name("generated_ep{:>3}.txt".format(epoch))

         gen_txt = "+++Example: +++\n"
         for prompt in self.prompts:
            prompt_token = self.__input_tokenize(prompt)
            # generate
            try:
               output_tokens = self.sampler(
                                 next=self.next,
                                 prompt=prompt_token,
                                 index=len(prompt)+1,
                              )
            except Exception as err:
               raise TypeError(f"Unexpected {err=}, {type(err)=}")
            # Decode the tokens back to text (useful for understanding)
            generated_txt = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            # save generated text
            gen_txt += "Generated prompt: @{}@\n\t{}\n".format(prompt, generated_txt)
         path_gen.write_text(gen_txt)

   def next(self, prompt, cache, index):
      logits = self.model(prompt)[:, index - 1, :]
      # Ignore hidden states for now; only needed for contrastive search.
      hidden_states = None
      return logits, hidden_states, cache
