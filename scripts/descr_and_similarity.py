import requests
from transformers import BertModel, BertTokenizer
import torch

import matplotlib.pyplot as plt
import seaborn as sns

def calculate_embeddings(dataset_name):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # Load BERT tokenizer and model
    model = BertModel.from_pretrained('bert-base-uncased')

    # List of words to encode
    if dataset_name=='cifar10' or dataset_name=='cifar100' or dataset_name=='lfw' or dataset_name=='ageDB':
        path = "data/"+dataset_name+"_classes.txt"
        classes = load_words_to_array(path)
    
    description=[]
    for y in classes:
        if y in ['aquarium_fish', 'lamp']:
            d = y
        else:
            d = get_wikipedia_description(y)
        if d == None or d == "":
            print(f"Could not retrieve description for {y}")
            description.append(y) # append the class name if no description is found
        else:
            description.append(d)
        
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
    with open(file_path, 'r') as f:
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


def calculate_dissimilarity(embeddings, device='cpu'):
    embeddings = embeddings.to(device)
    # Calculate the similarity matrix
    dissimilarity = 1 - torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    return dissimilarity

def visualize_dissimilarity_matrix(matrix, labels):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix.numpy(), xticklabels=labels, yticklabels=labels, cmap="coolwarm", annot=False, fmt=".2f")
    plt.title("Dissimilarity Matrix")
    plt.xlabel("Classes")
    plt.ylabel("Classes")
    plt.savefig("dissimilarity_matrix.png")

def main():
    dataset_name = "cifar100"
    embeddings = calculate_embeddings(dataset_name)  
    mean_embeddings = embeddings.mean(dim=1) 
    dissimilarity_matrix = calculate_dissimilarity(mean_embeddings) 
    
    class_labels = load_words_to_array(f"data/{dataset_name}_classes.txt")
    visualize_dissimilarity_matrix(dissimilarity_matrix, class_labels)

if __name__ == '__main__':
    main()
