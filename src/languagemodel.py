from transformers import GPT2Tokenizer, GPT2LMHeadModel


class LanguageModel:
    def __init__(self, model_name) -> None:

        if model_name == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=self.tokenizer.eos_token_id)
            self.model.eval().cuda()
            self.max_input_ids_length = 1022
        else:
            raise NotImplementedError(f"Currently, only 'model_name'=='gpt2' is supported, your input: {model_name}")

    def input_length_check(self, input_ids):
        input_ids_length = len(input_ids[0])
        if input_ids_length > self.max_input_ids_length:
            raise ValueError(
                f"Input sequence length can be max {self.max_input_ids_length} and is {len(input_ids[0])=} long"
            )

    def generate(self, prompt: str, max_predicted_tokens: int = 5, stop_token=None, do_sample=False):
        """
        Return as text the prompt plus the next 'max_predicted_tokens' predictions.
        If a stop_token is given, only return the text until the stop_token appears in the prediction.
        """
        # Encodes the prompt into token IDs
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        # input_ids_length = len(input_ids[0])
        self.input_length_check(input_ids)

        # switch to use 'max_new_tokens' instead of 'max_length', because it was more clear
        generated_text_ids = self.model.generate(
            input_ids=input_ids.cuda(), max_new_tokens=max_predicted_tokens, do_sample=do_sample
        )
        # The tokens seem to have a space in the beginning when they are not the first word in a sentence and no space if they are
        # When decoding, we want to remove these extra spaces
        generated_text = self.tokenizer.decode(generated_text_ids[0], clean_up_tokenization_spaces=True)
        post_prompt_text = generated_text[
            len(self.tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True)) :
        ]

        # Return the prompt + predicted answer and optionally remove everything predicted after a stop_token
        return (
            prompt + post_prompt_text[: post_prompt_text.find(stop_token) if stop_token else None]
        )  # , input_ids_length

    # Note that the logits are shifted over 1 to the left, since HuggingFace doesn't give a logit for the first token
    def get_logits_and_tokens(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        self.input_length_check(input_ids)
        tokens = [self.tokenizer.decode([input_id]) for input_id in input_ids[0]]
        # Running the complete text through the model again to get the raw logits
        # Output size example: (batch_size, sequence_length, vocab_size)
        output = self.model(input_ids.cuda())
        return output.logits[0], tokens

    @staticmethod
    def render_example(text, label, include_abstract=True):
        """
        Extracts and preprocesses title and abstract text into a string ready for being
        a labeled example for the prompt of a language model.
        """
        title = text.split(".")[0].strip()
        if include_abstract:
            abstract = text[len(title) + 1 :].strip()
            return f"""Title: {title}\nAbstract: {abstract}\nLabel: {label}"""
        else:
            return f"""Title: {title}\nLabel: {label}"""

    @staticmethod
    def render_end_example(text, include_abstract=True):
        """
        Extracts and preprocesses title and abstract text into a string ready for being
        the end example predicted for by a language model.
        """
        title = text.split(".")[0].strip()
        if include_abstract:
            abstract = text[len(title) + 1 :].strip()
            return f"""Title: {title}\nAbstract: {abstract}\nLabel:"""
        else:
            return f"""Title: {title}\nLabel:"""

    @staticmethod
    def make_prompt(instructions, prompt_texts, prompt_labels, end_text, include_abstract=True):
        """
        Creates a prompt by taking instructions, adding some training examples of input and
        expected output and ending with the input to be predicted for by the language model
        """
        rendered_train_examples = "\n\n--\n\n".join(
            [
                LanguageModel.render_example(text, label, include_abstract)
                for text, label in zip(prompt_texts, prompt_labels)
            ]
        )
        return f"""{instructions}\n\n{rendered_train_examples}\n\n--\n\n{LanguageModel.render_end_example(end_text, include_abstract)}"""
