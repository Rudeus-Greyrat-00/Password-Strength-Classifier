import torch
from torch import nn
from torch import optim
from torchtext.vocab import build_vocab_from_iterator
from utils import PswDataset, double_char_tokenizer
from torch.utils.data import DataLoader
from models import SimpleTextClassificationModel
import time
from training import evaluate, train

"""
https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = PswDataset("data.csv")
train_ds, test_ds = dataset.split(0.2)
train_ds, val_ds = train_ds.split(0.2)

tokenizer = double_char_tokenizer
train_iter = iter(train_ds)


def yield_tokens(data_iter):
    for text, label in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x)


def collate_batch(batch):
    """
    In this example, the text entries in the original data batch input are packed into a list and concatenated as a
    single tensor for the input of nn.EmbeddingBag. The offset is a tensor of delimiters to represent the beginning
    index of the individual sequence in the text tensor. Label is a tensor saving the labels of individual text entries
    :param batch:
    :return:
    """
    label_list, text_list, offsets = [], [], [0]
    for (_text, _label) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

"""
dataloader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_batch)
dataloader2 = DataLoader(train_ds, batch_size=64, shuffle=False, collate_fn=collate_batch)

for idx, (label, text, offsets) in enumerate(dataloader):
    print(label)
    break

for idx, (label, text, offsets) in enumerate(dataloader2):
    print(label)
    break

exit(0)
"""
num_classes = len(set(label for (label, text) in train_ds))
vocab_size = len(vocab)
emsize = 64  # embedding dimension
model = SimpleTextClassificationModel(vocab_size, emsize, num_classes).to(device)

# Hyperparameters
EPOCHS = 10 # epoch
LR = 5  # learning rate
BATCH_SIZE = 64 # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None

train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)

print("Starting training...")

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(dataloader=train_dataloader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch)
    accu_val = evaluate(dataloader=valid_dataloader, model=model , criterion=criterion)
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)

print('Checking the results of test dataset.')
accu_test = evaluate(dataloader=test_dataloader, model=model, criterion=criterion)
print('test accuracy {:8.3f}'.format(accu_test))



