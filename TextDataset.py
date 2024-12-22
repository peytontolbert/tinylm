from torch.utils.data import Dataset
class TextDataset(Dataset):
    def __init__(self, encoded_text, block_size):
        # Assuming encoded_text is a list of integers representing encoded characters
        self.encoded_text = encoded_text
        self.block_size = block_size

    def __len__(self):
        # The length is the number of blocks we can make
        return len(self.encoded_text) - self.block_size

    def __getitem__(self, idx):
        # Get the sequence of tokens that starts at this index
        chunk = self.encoded_text[idx:idx + self.block_size + 1]
        x = chunk[:-1].clone().detach()
        y = chunk[1:].clone().detach()
        return x, y
