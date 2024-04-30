# Required imports
import torch
from torch import nn
import torchvision
import json
import re
from torchvision.models.resnet import ResNet18_Weights, resnet18
from torchvision import transforms as tvt
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import pytorch_lightning as pl
from torchaudio.functional import edit_distance
from torchtext.data.metrics import bleu_score
from evaluate import load
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import os
import torch.multiprocessing as mp

# Ensure compatibility with MPS and avoid multiprocessing issues
torch.multiprocessing.set_sharing_strategy('file_system')

# Text processing for LaTeX formulas
class TextProcessor:
    SPECIAL_TOKENS = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}

    def __init__(self, vocab_file):
        with open(vocab_file, 'r') as file:
            self.vocab = json.load(file)
        
        self.pad_id = self.SPECIAL_TOKENS['<pad>']

        self.id2word = [None] * (len(self.vocab) + len(self.SPECIAL_TOKENS))
        for token, index in self.SPECIAL_TOKENS.items():
            self.id2word[index] = token
        offset = len(self.SPECIAL_TOKENS)
        for i, word in enumerate(self.vocab):
            if word not in self.SPECIAL_TOKENS.values():
                self.id2word[i + offset] = word
        
        self.word2id = {word: idx for idx, word in enumerate(self.id2word) if word is not None}
        self.vocab_size = len(self.id2word)

        # Correct the regex pattern
        self.tokenize_regex = re.compile(
            "(\\\\[a-zA-Z]+)|" + '((\\\\)*[$-/:-?{-~!"^_`\[\]])|' + "(\w)|" + "(\\\\)"
        )

    def split_into_tokens(self, formula: str):
        return str([m.group(0) for m in re.finditer(self.tokenize_regex, formula)])

    #text to Tensor of integers based on token IDs
    def encode(self, formula: str):
        tokens = self.split_into_tokens(formula)
        return [self.SPECIAL_TOKENS['<sos>']] + [self.word2id.get(token, self.SPECIAL_TOKENS['<unk>']) for token in tokens] + [self.SPECIAL_TOKENS['<eos>']]

    #Tensor of integers based on token IDs to text string
    def decode(self, token_ids):
        return ' '.join(self.id2word[i] for i in token_ids if i not in [self.SPECIAL_TOKENS['<pad>'], self.SPECIAL_TOKENS['<sos>'], self.SPECIAL_TOKENS['<eos>']])

        
# Dataset class for handling LaTeX images and formulas
class LatexDataset(Dataset):
    def __init__(self, data_path, img_path, data_type: str, text_processor: TextProcessor):
        #assert data_type in ["train", "test", "validate"], "Invalid data type specified"
        assert data_type in ["train", "test", "validate"], "Invalid data type specified"
        self.data_frame = pd.read_excel(f"{data_path}/im2latex_{data_type}.xlsx")
        self.data_frame["image"] = self.data_frame["image"].apply(lambda x: f"{img_path}/{x}")
        self.transform = tvt.Compose([
            #tvt.Grayscale(num_output_channels=3),
            tvt.Grayscale(num_output_channels=1),
            tvt.ConvertImageDtype(torch.float),  # Convert image to float tensor
            tvt.Normalize(mean=[0.5], std=[0.5])  # Adjust normalization for single-channel input
        ])
        self.text_processor = text_processor

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        image_path = row["image"]
        image = torchvision.io.read_image(image_path)
        image = self.transform(image)
        formula = row["formula"]
        encoded_formula = torch.tensor(self.text_processor.encode(formula), dtype=torch.long)
        return image, encoded_formula


# Convolutional Encoder 
class ConvEncoder(nn.Module):
    def __init__(self, enc_dim):
        super(ConvEncoder, self).__init__()
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # Pooling to manage one dimension specifically
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # Pooling to manage the other dimension
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, enc_dim, 3, padding=1),  # Adjusting to the desired encoding dimension
            nn.BatchNorm2d(enc_dim),
        )
        self.enc_dim = enc_dim
    
    def forward(self, x):
        print(x.shape)
        x = self.feature_encoder(x)
        # Permutes and reshapes output for sequence processing compatibility
        print(f"Encoder output shape before permutation: {x.shape}")
        x = x.permute(0, 2, 3, 1)
        bs, _, _, d = x.size()
        x = x.view(bs, -1, d)
        #x = x.permute(2, 3, 1)
        return x


# Transformer Decoder for processing encoded images into LaTeX formulas
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead), num_layers=6
        )

    def forward(self, tgt, memory):
        # Embedding and initial shape logging
        tgt = self.embedding(tgt)
        #print(f"Initial tgt shape: {tgt.shape}")
        tgt = tgt.permute(1, 0, 2)  # Correctly reshaping to [seq_len, batch_size, features]
        #print(f"Decoder input shape: {tgt.shape}")

        # Correct permutation for memory to match Transformer expectations
        memory = memory.permute(1, 0, 2)  # seq_length, batch_size, features
        #print(f"Memory input shape after permutation: {memory.shape}")

        # Pass to Transformer decoder
        output = self.transformer_decoder(tgt, memory)
        return output


# Main Image2Latex Model combining the encoder and decoder
class Image2LatexModel(pl.LightningModule):
    def __init__(self, encoder, decoder, text, sos_id, eos_id, max_length=150):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = nn.CrossEntropyLoss()
        self.text = text
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = text.pad_id
        self.max_length = max_length
        self.exact_match = load("exact_match")

    def forward(self, images, formulas):
        encoder_outputs = self.encoder(images)
        print(f"Encoder output shape: {encoder_outputs.shape}")
        formulas_input = formulas[:, :-1]  # Assuming you want to drop the last token
        print(f"Prepared formulas input shape: {formulas_input.shape}")
        decoder_outputs = self.decoder(formulas_input, encoder_outputs)
        return decoder_outputs

    def training_step(self, batch, batch_idx):
        images, formulas = batch
        print(formulas.shape)
        formulas_input = formulas[:, :-1]
        formulas_target = formulas[:, 1:-1]
        output = self(images, formulas_input)

        # Log shape before adjustment
        print(f"Pre-adjustment output shape: {output.shape}")
        
        # # Adjust the output shape
        # output_adjusted = output[:, :formulas_target.shape[1], :]
        # print(f"Training output adjusted shape: {output_adjusted.shape}")
        # print(f"Training target shape: {formulas_target.shape}")

        # loss = self.loss_fn(output_adjusted.transpose(1, 2), formulas_target)
        # print(f"Training Loss: {loss.item()}")  # Print the loss value for immediate feedback
        # self.log('train_loss', loss)
        # return loss
    
        #Ensure output is correctly shaped for loss calculation: [batch_size, num_classes, seq_len]
        output = output.permute(1, 2, 0)  # Change from [seq_len, batch_size, num_classes] to [batch_size, num_classes, seq_len]
        output_adjusted = output[:, :, :formulas_target.shape[1]]  # Ensure the sequence length matches that of the target

        print(f"Training output adjusted shape: {output_adjusted.shape}")
        print(f"Training target shape: {formulas_target.shape}")

        loss = self.loss_fn(output_adjusted, formulas_target)
        print(f"Training Loss: {loss.item()}")  # Print the loss value for immediate feedback
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, formulas = batch
        formulas_input = formulas[:, :-1]
        formulas_target = formulas[:, 1:]  # Correctly matching the shifted output for loss computation

        output = self(images, formulas_input)

        # Assuming output is [seq_len, batch_size, num_classes] and target is [batch_size, seq_len]
        output_adjusted = output.transpose(0, 1)  # Change to [batch_size, seq_len, num_classes]
        output_adjusted = output_adjusted[:, :formulas_target.size(1), :]  # Adjust seq_len if necessary
        formulas_target = formulas_target[:, :output_adjusted.size(1)]
        print(f"Output adjusted shape: {output_adjusted.shape}")
        print(f"Target shape: {formulas_target.shape}")
        loss = self.loss_fn(output_adjusted.flatten(0, 1), formulas_target.flatten())  # Flatten for CrossEntropyLoss if required


        # Transpose output for loss calculation: [batch_size, num_classes, seq_len]
        loss = self.loss_fn(output_adjusted.transpose(1, 2), formulas_target)
        print(f"Validation Loss: {loss.item()}")  # Print the loss value for immediate feedback

        # Decode outputs to text for further metric evaluation
        output_texts = [self.decode(output_adjusted[i]) for i in range(output_adjusted.shape[0])]
        predicts = [self.text.split_into_tokens(text) for text in output_texts]
        #predicts = [self.text.split_into_tokens(self.decode(output[i])) for i in range(output_adjusted.shape[0])]
        truths = [self.text.split_into_tokens(self.text.decode(formulas_target[i])) for i in range(formulas_target.shape[0])]

        # for predict, truth in zip(predicts, truths):
        #     print(f"Predicted: {' '.join(predict)}, Ground Truth: {' '.join(truth)}")

        # Calculate additional metrics
        #edit_dist = torch.mean(torch.Tensor([edit_distance(torch.tensor(pre), torch.tensor(tru)) for pre, tru in zip(predicts, truths)]))
        #bleu4 = torch.mean(torch.Tensor([bleu_score([pre], [[tru]]) for pre, tru in zip(predicts, truths)]))
        #em = torch.mean(torch.Tensor([self.exact_match.compute(predictions=[" ".join(pre)], references=[" ".join(tru)])['exact_match'] for pre, tru in zip(predicts, truths)]))

        self.log('val_loss', loss)
        # self.log('val_edit_distance', edit_dist)
        # self.log('val_bleu4', bleu4)
        # self.log('val_exact_match', em)

        #return {"val_loss": loss, "val_edit_distance": edit_dist, "val_bleu4": bleu4, "val_exact_match": em}
        return {"val_loss": loss}


    def test_step(self, batch, batch_idx):
        images, formulas = batch
        formulas_input = formulas[:, :-1]
        formulas_target = formulas[:, 1:]  # Correctly matching the shifted output for loss computation

        output = self(images, formulas_input)

        # Assuming output is [seq_len, batch_size, num_classes] and target is [batch_size, seq_len]
        output_adjusted = output.transpose(0, 1)  # Change to [batch_size, seq_len, num_classes]
        output_adjusted = output_adjusted[:, :formulas_target.size(1), :]  # Adjust seq_len if necessary
        formulas_target = formulas_target[:, :output_adjusted.size(1)]
        #print(f"Output adjusted shape: {output_adjusted.shape}")
        #print(f"Target shape: {formulas_target.shape}")
        loss = self.loss_fn(output_adjusted.flatten(0, 1), formulas_target.flatten())  # Flatten for CrossEntropyLoss if required


        # Transpose output for loss calculation: [batch_size, num_classes, seq_len]
        loss = self.loss_fn(output_adjusted.transpose(1, 2), formulas_target)
        print(f"Test Loss: {loss.item()}")  # Print the loss value for immediate feedback

        # Decode outputs to text for further metric evaluation
        output_texts = [self.decode(output_adjusted[i]) for i in range(output_adjusted.shape[0])]
        predicts = [self.text.split_into_tokens(text) for text in output_texts]
        truths = [self.text.split_into_tokens(self.text.decode(formulas_target[i])) for i in range(formulas_target.shape[0])]

        for predict, truth in zip(predicts, truths):
             print(f"Predicted: {' '.join(predict)}, Ground Truth: {' '.join(truth)}")

        # Calculate additional metrics
        edit_dist = torch.mean(torch.Tensor([edit_distance(torch.tensor(pre), torch.tensor(tru)) for pre, tru in zip(predicts, truths)]))
        bleu4 = torch.mean(torch.Tensor([bleu_score([pre], [[tru]]) for pre, tru in zip(predicts, truths)]))
        em = torch.mean(torch.Tensor([self.exact_match.compute(predictions=[" ".join(pre)], references=[" ".join(tru)])['exact_match'] for pre, tru in zip(predicts, truths)]))

        self.log('test_loss', loss)
        self.log('val_edit_distance', edit_dist)
        self.log('val_bleu4', bleu4)
        self.log('val_exact_match', em)

        #return {"val_loss": loss, "val_edit_distance": edit_dist, "val_bleu4": bleu4, "val_exact_match": em}
        return {"test_loss": loss}

    def decode(self, x):
        
        # Get the encoder outputs
        temp = x.unsqueeze(0)
        memory = self.encoder(temp.unsqueeze(0)) 

        # Start the sequence with the <sos> token
        tgt_input = torch.tensor([[self.sos_id]], dtype=torch.long, device=x.device)
        
        predictions = []
        for _ in range(self.max_length):
            output = self.decoder(tgt_input, memory)
            temp = output.argmax(-1)
            # extract the elements of tensor
            next_token = output.argmax(-1)[0, -1].item()  # Get the last token from the sequence
            predictions.append(next_token)

            # Break if <eos> token is generated
            if next_token == self.eos_id:
                break

            # Append the predicted token to the sequence and continue
            tgt_input = torch.cat([tgt_input, torch.tensor([[next_token]], device=x.device)], dim=1)
        
        # Convert the list of token indices to text using the provided text processing utility
        #temp = str(predictions)
        #decoded_text = self.text.encode(temp)
        # Convert the list of token indices to text using the provided text processing utility
        decoded_text = ' '.join(self.text.id2word[idx] for idx in predictions if idx not in [self.text.pad_id, self.sos_id, self.eos_id])
        return decoded_text

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

# DataModule for handling data loading
class LatexDataModule(pl.LightningDataModule):
    def __init__(self, data_path, img_path, batch_size=16, num_workers=4, text_processor=None):
        super().__init__()
        self.data_path = data_path
        self.img_path = img_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.text_processor = text_processor

    def setup(self, stage=None):
        #self.train_dataset = LatexDataset(self.data_path, self.img_path, "train", self.text_processor)
        self.train_dataset = LatexDataset(self.data_path, self.img_path, "train", self.text_processor)
        print(f"Train dataset is working")
        self.val_dataset = LatexDataset(self.data_path, self.img_path, "validate", self.text_processor)
        self.test_dataset = LatexDataset(self.data_path, self.img_path, "test", self.text_processor)
    
    def collate_fn(self, batch):
        images, formulas = zip(*batch)
        
        # Determine the maximum width and height in the batch
        max_width = max(img.shape[2] for img in images)
        max_height = max(img.shape[1] for img in images)
        
        # Pad images to the maximum width and height
        padded_images = []
        for img in images:
            # Calculate padding
            left = (max_width - img.shape[2]) // 2
            right = max_width - img.shape[2] - left
            top = (max_height - img.shape[1]) // 2
            bottom = max_height - img.shape[1] - top
            
            # Apply padding
            padded_img = torch.nn.functional.pad(img, (left, right, top, bottom), "constant", 0)
            padded_images.append(padded_img)
        
        images = torch.stack(padded_images, dim=0)
        
        formula_lengths = [len(formula) for formula in formulas]
        max_length = max(formula_lengths)
        padded_formulas = [torch.cat((formula, torch.tensor([self.text_processor.pad_id] * (max_length - len(formula)), dtype=torch.long))) for formula in formulas]
        padded_formulas_tensor = torch.stack(padded_formulas, dim=0)
        
        print(f"Batched images shape: {images.shape}")
        print(f"Batched formulas shape: {padded_formulas_tensor.shape}")
    
        return images, padded_formulas_tensor

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn)

# Main function to orchestrate the training process
def main():
    # Define paths and hyperparameters
    data_path = 'Final_Project/im2latex_sorted_by_size'
    img_path = 'Final_Project/formula_images_processed'
    #batch_size = 16
    batch_size = 16
    vocab_file = 'Final_Project/100k_vocab.json'
    text_processor = TextProcessor(vocab_file)
    vocab_size = text_processor.vocab_size

    print(f"Vocabulary size: {vocab_size}")

    # Initialize components
    encoder = ConvEncoder(enc_dim=512)
    decoder = TransformerDecoder(vocab_size=text_processor.vocab_size, d_model=512, nhead=8)
    model = Image2LatexModel(encoder=encoder, decoder=decoder, text=text_processor, 
                             sos_id=TextProcessor.SPECIAL_TOKENS['<sos>'],
                             eos_id=TextProcessor.SPECIAL_TOKENS['<eos>'], max_length=150)
    data_module = LatexDataModule(data_path, img_path, batch_size=batch_size, text_processor=text_processor)

    # Initialize loggers
    tensorboard_logger = TensorBoardLogger("tb_logs", name="my_model")
    csv_logger = CSVLogger("logs", name="my_model")

    # Initialize a trainer with both loggers
    checkpoint_callback = ModelCheckpoint(dirpath="./checkpoints", save_top_k=1, monitor="val_loss")
    trainer = pl.Trainer(max_epochs=2, callbacks=[checkpoint_callback], logger=[tensorboard_logger, csv_logger])

    # Train the model
    trainer.fit(model, data_module)

    trainer.validate(model, data_module)
    
    trainer.test(model, data_module)


if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) is None and os.name != 'nt':
        mp.set_start_method('fork')
    main()


