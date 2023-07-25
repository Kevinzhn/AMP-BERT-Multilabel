import os
import pandas as pd
import numpy as np
import torch
import re

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, hamming_loss, jaccard_score
from torch.utils.data import Dataset, DataLoader

# Verificar se está utilizando GPU
torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from transformers import AutoTokenizer, Trainer, TrainingArguments, BertForSequenceClassification, AdamW


# Definir a estrutura de dados que vai receber do database
class amp_data(Dataset):
    def __init__(self, df, tokenizer_name='Rostlab/prot_bert_bfd', max_len=200):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        self.max_len = max_len
        self.seqs, self.labels = self.get_seqs_labels(df)
      
    # Separar a sequência e os tipos de proteína
    def get_seqs_labels(self, df):
        seqs = list(df['Sequence'])
        labels = list(df[['Antibacterial', 'Antiviral', 'Antiparasitic', 'Antifungal']].values)
        labels = torch.tensor(labels, dtype=torch.float32)
        return seqs, labels
      
    # Obter a quantidade da dados do database
    def __len__(self):
        return len(self.labels)
      
    # Formatar o dado para o treinamento
    def __getitem__(self, idx):
        seq = " ".join("".join(self.seqs[idx].split()))
        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_len)

        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample['labels'] = torch.tensor(self.labels[idx])

        return sample

# Ler o conjunto de dados do treinamento
data_url = 'https://raw.githubusercontent.com/Kevinzhn/AMP-BERT-Multilabel/main/treinamento'
df = pd.read_csv(data_url, index_col=None)  # Ignorar a coluna "numero"
df = df.sample(frac=1, random_state=0)
print(df.head(7))
train_dataset = amp_data(df)

# Definir as métricas necessárias para avaliação de desempenho
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions > 0.5
    hamming_loss_val = hamming_loss(labels, preds)
    jaccard_score_val = jaccard_score(labels, preds, average='samples') 
    return {
        'hamming_loss': hamming_loss_val,
        'jaccard_score_samples': jaccard_score_val,
    }

# Definir a função de inicialização do modelo para o Trainer no Huggingface
def model_init():
    return BertForSequenceClassification.from_pretrained('Rostlab/prot_bert_bfd', num_labels=4)

# Ler o conjunto de dados de avaliação
eval_data_url = 'https://raw.githubusercontent.com/Kevinzhn/AMP-BERT-Multilabel/main/teste' 
eval_df = pd.read_csv(eval_data_url, index_col=None)
eval_df = eval_df.sample(frac=1, random_state=0)
eval_dataset = amp_data(eval_df)

# Configuração do treinamento de modelo
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=15,  # Número de iterações
    learning_rate=5e-5,  # Taxa de aprendizagem
    per_device_train_batch_size=1,
    warmup_steps=0,
    weight_decay=0.1,
    logging_dir='./logs',
    logging_steps=100,
    do_train=True,
    do_eval=True,  # Aplicar avaliação de desempenho
    evaluation_strategy="epoch",  # Avaliação de desempenho a cada iteração
    save_strategy='epoch',  # Salvar o modelo a cada iteração
    gradient_accumulation_steps=64,
    fp16=True,
    fp16_opt_level="O2",
    run_name="AMP-BERT",
    seed=0,
    load_best_model_at_end=True
)

# Configuração da incialização
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
