import json

class FlatForgetDataset:
    def __init__(self, path):
        self.data = []
        with open(path) as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return ex["prompt"], ex["forget"], ex["template"]


