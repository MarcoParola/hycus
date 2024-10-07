from torch.utils.data import Dataset
from src.models.resnet import retrieve_weights

class DatasetWrapperIcus(Dataset):
    def __init__(self, infgt, cuda=False, model, orig_dataset="cifar10"):
        """
        Args:
            classi (Tensor): Tensor contenente le etichette delle classi.
            pesi (Tensor): Tensor contenente i pesi associati a ciascuna classe.
            descrizioni (Tensor): Tensor contenente le descrizioni (o embedding) delle classi.
            infgt_flags (Tensor): Tensor contenente i flag infgt per ciascuna classe (0 o 1).
            cuda (bool): Se True, carica i dati su GPU.
        """
        if orig_dataset == 'cifar10':
            classes=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        self.classes = classes  # Tensor delle classi
        self.descr = calculate_embeddings(orig_dataset)  # Tensor delle descrizioni
        self.weights, self.bias=retrieve_weights(model)
        self.infgt = infgt  # Tensor dei flag infgt (1 o 0)
        self.cuda = cuda
        self.orig_dataset = orig_dataset

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        # Estrai la classe, i pesi, la descrizione e il flag infgt in base all'indice
        classe = self.classes[idx]
        weigths = self.weigths[idx]
        descr = self.descr[idx]
        infgt = self.infgt[idx]

        # Se richiesto, trasferisci su GPU
        if self.cuda:
            classe = classe.cuda()
            weigths = weigths.cuda()
            descr = descr.cuda()
            infgt = infgt.cuda()

        # Restituisce la tupla (classe, pesi, descrizione, infgt)
        return classe, weigths, descr, infgt



    def calculate_embeddings(self, dataset_name):
        # Set random seed for reproducibility
        RandomSeed = 52
        random.seed(RandomSeed)
        torch.manual_seed(RandomSeed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RandomSeed)

        # Load BERT tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        # List of words to encode
        if dataset_name=='cifar10':
            text = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

        # Tokenize the list of words all together
        encoding = tokenizer.batch_encode_plus(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        )

        # Get token IDs and attention mask
        token_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Display token IDs and attention mask
        print(f"Token IDs: {token_ids}")
        print(f"Attention Mask: {attention_mask}")

        # Get word embeddings for all the words at once
        with torch.no_grad():
            outputs = model(input_ids=token_ids, attention_mask=attention_mask)
            word_embeddings = outputs.last_hidden_state  # Retrieve the last hidden states

        # Display the shape of the word embeddings
        print(f"Word Embeddings Shape: {word_embeddings.shape}")
        return word_embeddings
        
    
                

