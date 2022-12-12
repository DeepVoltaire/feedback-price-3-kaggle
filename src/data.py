import torch
from torch.utils.data import Dataset


class FeedbackDataset(Dataset):
    def __init__(
        self,
        targets,
        texts,
        tokenizer,
        max_length,
        dataset_length_multiplier: int = 1,
        submit: bool = False,
    ):
        # if dataset_length_multiplier != 1:
        #     print(
        #         f"{len(texts)} sequences repeated {dataset_length_multiplier} times = "
        #         f"{len(texts) * dataset_length_multiplier} dataset length"
        #     )
        self.targets = targets  # * dataset_length_multiplier
        self.texts = texts  # * dataset_length_multiplier
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.submit = submit

    def prepare_inputs(self, abstract):
        inputs = self.tokenizer.encode_plus(
            abstract,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.prepare_inputs(self.texts[idx])

        if self.submit:
            return inputs
        else:
            label = torch.tensor(self.targets[idx], dtype=torch.float)
            return inputs, label
