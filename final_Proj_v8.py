import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import pytorch_lightning as pl
from torchvision import transforms as tvt
from torchvision import io as tio
from transformers import GPT2Tokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.models import resnet18, ResNet18_Weights
from torchmetrics import Accuracy


# Define the dataset for handling LaTeX images and formulas
class LatexDataset(Dataset):
    def __init__(self, data_path, img_path, data_type, subsample):
        self.data_frame = pd.read_excel(f"{data_path}/im2latex_{data_type}.xlsx")
        self.data_frame["image"] = self.data_frame["image"].apply(lambda x: f"{img_path}/{x}")
        # If subsampling is requested, randomly select a subset of the dataframe
        if subsample is not None:
            self.data_frame = self.data_frame.sample(n=subsample, random_state=42).reset_index(drop=True)

        self.transform = tvt.Compose([
            tvt.Resize((224, 224)),  # Adjust the size to fit ResNet input requirements
            tvt.ConvertImageDtype(torch.float),  # Ensure dtype is correct
            tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        image_path = row["image"]
        image = tio.read_image(image_path)
        image = self.transform(image)
        formula = row["formula"]
        return image, formula

# Use a pre-trained ResNet as the encoder
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        weights = ResNet18_Weights.DEFAULT  # Use the most up-to-date default weights
        base_model = resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

# Inside your TextDecoder class, redefine the Accuracy instantiation:
class TextDecoder(pl.LightningModule):
    def __init__(self, encoder, hidden_dim, vocab_file):
        super(TextDecoder, self).__init__()
        self.encoder = encoder
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids('<pad>')

        with open(vocab_file, "r") as f:
            custom_vocab = eval(f.read())
        self.tokenizer.add_tokens(custom_vocab)

        self.embedding = nn.Embedding(len(self.tokenizer), hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, len(self.tokenizer))
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # Correctly instantiate the Accuracy metric with task and num_classes specified
        self.accuracy = Accuracy(task="multiclass", num_classes=len(self.tokenizer))

    def forward(self, images, captions):
        features = self.encoder(images)
        inputs = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)
        embedded = self.embedding(inputs)
        lstm_out, _ = self.lstm(embedded, (features.unsqueeze(0), torch.zeros_like(features).unsqueeze(0)))
        outputs = self.fc(lstm_out)

        # Prepare outputs for accuracy computation
        preds = outputs.argmax(dim=-1)
        return outputs, preds, inputs

    def _step(self, batch):
        images, captions = batch
        outputs, preds, labels = self(images, captions)
        loss = self.loss_fn(outputs.transpose(1, 2), labels)
        acc = self.accuracy(preds, labels)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log('train_loss', loss)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log('val_loss', loss)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-1)

# DataModule to manage training and validation data loaders
class LatexDataModule(pl.LightningDataModule):
    def __init__(self, data_path, img_path, batch_size=16, num_workers=11, subsampleTrain=500, subsampleVal=50, subsampleTest=50):
        super().__init__()
        self.data_path = data_path
        self.img_path = img_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subsampleTrain = subsampleTrain
        self.subsampleVal = subsampleVal
        self.subsampleTest = subsampleTest

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = LatexDataset(self.data_path, self.img_path, 'train', self.subsampleTrain)
            self.val_dataset = LatexDataset(self.data_path, self.img_path, 'validate', self.subsampleVal)
        if stage == 'test' or stage is None:
            self.test_dataset = LatexDataset(self.data_path, self.img_path, 'test', self.subsampleTest)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

# Train the model
def train_model():
    data_path = 'Final_Project/im2latex_sorted_by_size'
    img_path = 'Final_Project/formula_images_processed'
    batch_size = 16
    num_workers = 11

    encoder = ImageEncoder()
    vocab_file = 'Final_Project/100k_vocab.json'
    hidden_dim = 512  # Example dimension for LSTM

    data_module = LatexDataModule(data_path, img_path, batch_size, num_workers, subsampleTrain=500, subsampleVal=50, subsampleTest=50)
    model = TextDecoder(encoder, hidden_dim, vocab_file)

    # Setup checkpoints and trainer
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', filename='best-checkpoint', save_top_k=1, mode='min')
    trainer = pl.Trainer(max_epochs=5, log_every_n_steps=5, callbacks=[checkpoint_callback])
    trainer.fit(model, data_module)
    checkpoint_callback.best_model_path
    trainer.validate(model, data_module)
    trainer.test(model, data_module)

if __name__ == "__main__":
    train_model()