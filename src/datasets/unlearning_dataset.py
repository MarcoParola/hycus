import torch
from torch.utils.data import Dataset
import random
#from src.utils import retrieve_weights
import requests
from transformers import BertModel, BertTokenizer

class UnlearningDataset(Dataset):
    def __init__(self, dataset, forget_indices):
        """
        UnlearningDataset class.
        Args:
            dataset (Dataset): original image dataset.
            forget_indices (list): index of the samples to forget.
        """
        self.dataset = dataset
        self.forget_indices = set(forget_indices) 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, label = self.dataset[index]
        infgt = 1 if index in self.forget_indices else 0
        return input, label, infgt


class IcusUnlearningDataset(Dataset):
    def __init__(self, orig_dataset, nlayers, infgt, model, num_classes, device="cpu"):
        """
        IcusUnlearningDataset class.
        Args:
            orig_dataset (str): name of the original dataset.
            infgt (Tensor): flag infgt (1 or 0).
            model (nn.Module): model to use.
            num_classes (int): number of classes.
            device (str): device to use.
        """
        self.classes = torch.arange(0, num_classes)
        self.descr = self.calculate_embeddings(orig_dataset)  # Tensor delle descrizioni
        self.distinct, self.shared = model.get_weights(num_classes, nlayers)  # Tensor dei pesi distinti e condivisi
        self.infgt = infgt  # Tensor dei flag infgt (1 o 0)
        self.device = device

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        # Estrai la classe, i pesi, la descrizione e il flag infgt in base all'indice
        classe = self.classes[idx].to(self.device)
        if len(self.distinct) == 0:
            weigths = self.shared.to(self.device)
        else:
            weigths = torch.cat((self.distinct[idx], self.shared), 0).to(self.device)
        descr = self.descr[idx].to(self.device)
        infgt = self.infgt[idx].to(self.device)
        return classe, weigths, descr, infgt


    def calculate_embeddings(self, dataset_name):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # Load BERT tokenizer and model
        model = BertModel.from_pretrained('bert-base-uncased')

        # List of words to encode
        if dataset_name=='cifar10' or dataset_name=='cifar100':
            path = "data/"+dataset_name+"_classes.txt"
            classes = load_words_to_array(path)
        
        description=[]
        for y in classes:
            y = get_wikipedia_description(y)
            description.append(y)
            
        # Tokenize the list of words all together
        encoding = tokenizer.batch_encode_plus(
            description,
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True)

        # Get token IDs and attention mask
        token_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Get word embeddings for all the words at once
        with torch.no_grad():
            outputs = model(input_ids=token_ids, attention_mask=attention_mask)
            word_embeddings = outputs.last_hidden_state  # Retrieve the last hidden states

        return word_embeddings
        
def load_words_to_array(file_path):
    # Leggi le parole dal file di testo
    with open(file_path, 'r') as f:
        # Rimuovi eventuali spazi bianchi e newline, e crea una lista di parole
        words = [line.strip() for line in f if line.strip()]
    return words



#function to retrieve a description from wikipedia given a name
def get_wikipedia_description(name, language="en"):
    url = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": name,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Check for HTTP request errors
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()))  # Get the first (and usually only) page
        
        if "extract" in page:
            return page["extract"]
        elif "missing" in page:
            return f"No Wikipedia page found for '{name}'."
        else:
            return "Unexpected error occurred."
    except requests.RequestException as e:
        return f"An error occurred while accessing the Wikipedia API: {e}"

    


def get_unlearning_dataset(cfg, unlearning_method_name, model, train, retain_indices, forget_indices, forgetting_subset): 
    if unlearning_method_name == 'icus':
        num_classes = cfg.dataset.classes
        infgt = torch.tensor([1 if i in forgetting_subset else 0 for i in range(len(train))])  
        unlearning_train = IcusUnlearningDataset(cfg.dataset.name, cfg.unlearn.nlayers, infgt, model, num_classes, cfg.device)
        unlearning_train = torch.utils.data.DataLoader(unlearning_train, batch_size=cfg.dataset.classes, num_workers=0)
    else:
        unlearning_train = UnlearningDataset(train, forget_indices)
        unlearning_train = torch.utils.data.DataLoader(unlearning_train, batch_size=cfg.train.batch_size, num_workers=8)
    return unlearning_train


description = []
x=load_words_to_array("data/cifar10_classes.txt")
for y in x:
    y = get_wikipedia_description(y)
    description.append(y)
