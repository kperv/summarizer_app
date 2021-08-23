class Model(pl.LightningModule):
  def __init__(self):
    super.__init__()
    self.l1 = nn.Linear(x, y)

  def forward(self, x):
    return torch.relu()

  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = F.pairwise_distance(y_hat, y)
    self.log("train_loss", loss, on_epoch=True)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model(x)
    loss = F.pairwise_distance(y_hat, y)
    self.log("val_loss", loss, on_epoch=True)

  def test_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model(x)
    loss = F.pairwise_distance(y_hat, y)
    self.log("test_loss", loss, on_epoch=True)


  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.lr)



class MlSumDataModule(pl.LightningDataModule):
  def __init__(self, batch_size=32):
    super.__init__()
    self.batch_size = batch_size

  def prepare_data(self):
    self.dataset_ru = load_dataset("mlsum", "ru")
    self.dataset_es = load_dataset("mlsum", "es")
    nlp_ru = spacy.load("ru_core_news_md")
    nlp_es = spacy.load("es_core_news_md")

  def train_dataloader(self):
    dataset_ru_train = DataLoader(
        self.dataset_ru["train"],
        batch_size=self.batch_size)
    dataset_es_train = DataLoader(
        self.dataset_es["train"],
        batch_size=self.batch_size)
    loaders = [dataset_ru_train, dataset_es_train]
    return train_loaders

  def val_dataloader(self):
    dataset_ru_val = DataLoader(
        self.dataset_ru["val"],
        batch_size=self.batch_size)
    dataset_es_val = DataLoader(
        self.dataset_es["val"],
        batch_size=self.batch_size)
    loaders = [dataset_ru_val, dataset_es_val]
    return val_loaders

  def test_dataloader(self):
    dataset_ru_test = DataLoader(
        self.dataset_ru["test"],
        batch_size=self.batch_size)
    dataset_es_test = DataLoader(
        self.dataset_es["test"],
        batch_size=self.batch_size)
    loaders = [dataset_ru_test, dataset_es_test]
    return test_loaders



# or call with pretrained model
model = MyLightningModule.load_from_checkpoint(PATH)
trainer = pl.Trainer()
trainer.test(model, dataloaders=test_dataloader)


class SumDataset(Dataset):
    """ Get information from the dataset[split] """


def __init__(self, dataset):
    self.dataset = dataset


def __len__(self):
    return len(self.dataset)


def __getitem__(self, index):
    text = self.dataset[index]['text']
    summary = self.dataset[index]['summary']
    return {text: text, summary: summary}


def get_ntexts_for_review(self, n):
    df = pd.DataFrame(self.dataset)
    return df['text'].sample(n, random_state=1).values