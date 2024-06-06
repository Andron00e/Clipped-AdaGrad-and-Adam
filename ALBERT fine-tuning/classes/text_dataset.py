from torch.utils.data import Dataset


class TextDatasetCoLa(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.x_data = tokenizer(
            dataframe["sentence"].to_numpy().tolist(),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.y_data = dataframe["label"].to_numpy()
        self.n_samples = dataframe.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        x_sample = {
            "input_ids": self.x_data["input_ids"][index],
            "token_type_ids": self.x_data["token_type_ids"][index],
            "attention_mask": self.x_data["attention_mask"][index],
        }
        y_sample = self.y_data[index]
        return x_sample, y_sample


class TextDatasetRTE(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.x_data = tokenizer(
            dataframe["sentence1"].to_numpy().tolist(),
            dataframe["sentence2"].to_numpy().tolist(),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.y_data = dataframe["label"].to_numpy()
        self.n_samples = dataframe.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        x_sample = {
            "input_ids": self.x_data["input_ids"][index],
            "token_type_ids": self.x_data["token_type_ids"][index],
            "attention_mask": self.x_data["attention_mask"][index],
        }
        y_sample = self.y_data[index]
        return x_sample, y_sample
