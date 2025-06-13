import torch
from torch.utils.data import Dataset

class TextDatasetCoLa(Dataset):
    def __init__(self, dataframe, tokenizer, max_length: int = 128):
        enc = tokenizer(
            dataframe["sentence"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_token_type_ids=True, 
        )

        if "token_type_ids" not in enc:
            batch_size, seq_len = enc["input_ids"].shape
            enc["token_type_ids"] = torch.zeros((batch_size, seq_len), dtype=torch.long)

        self.encodings = enc
        self.labels = torch.tensor(dataframe["label"].to_list(), dtype=torch.long)

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return (
            {
                "input_ids":       self.encodings["input_ids"][idx],
                "attention_mask":  self.encodings["attention_mask"][idx],
                "token_type_ids":  self.encodings["token_type_ids"][idx],
            },
            self.labels[idx],
        )


class TextDatasetRTE(Dataset):
    def __init__(self, dataframe, tokenizer, max_length: int = 128):
        enc = tokenizer(
            dataframe["sentence1"].tolist(),
            dataframe["sentence2"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_token_type_ids=True,
        )

        if "token_type_ids" not in enc:
            batch_size, seq_len = enc["input_ids"].shape
            enc["token_type_ids"] = torch.zeros((batch_size, seq_len), dtype=torch.long)

        self.encodings = enc
        self.labels = torch.tensor(dataframe["label"].to_list(), dtype=torch.long)

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return (
            {
                "input_ids":       self.encodings["input_ids"][idx],
                "attention_mask":  self.encodings["attention_mask"][idx],
                "token_type_ids":  self.encodings["token_type_ids"][idx],
            },
            self.labels[idx],
        )
