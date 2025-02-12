import zipfile
import os
from PIL import Image
from collections import defaultdict

def retrieve_AgeDB_dataset(zip_path, extract_folder):
    # Extract ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    # Find if there are subfolders
    subfolders = [f for f in os.listdir(extract_folder) if os.path.isdir(os.path.join(extract_folder, f))]
    if subfolders:
        extract_folder = os.path.join(extract_folder, subfolders[0])  

    # Images list
    file_list = [f for f in os.listdir(extract_folder) if f.endswith(".jpg")]

    # Dictionary to map class names to IDs
    name_to_id = {}
    class_counts = defaultdict(int)
   
    # Count the number of samples for each class
    for file in file_list:
        parts = file.split("_")
        if len(parts) < 4:
            continue  
        nome = parts[1].lower()  # Normalize class name
        class_counts[nome] += 1
    
    # Filter classes with less than 15 samples
    valid_classes = {name for name, count in class_counts.items() if count >= 15}

    dataset = []

    # New dataset with filtered classes
    for file in file_list:
        parts = file.split("_")
        if len(parts) < 4:
            continue

        id_foto, nome, età, sesso = parts[:4]  
        nome = nome.lower()

        if nome not in valid_classes:
            continue  # Skip invalid classes

        if nome not in name_to_id:
            name_to_id[nome] = len(name_to_id)  

        img_path = os.path.join(extract_folder, file)
        
        try:
            img = Image.open(img_path).convert("RGB")  
            dataset.append((img, name_to_id[nome]))  
        except Exception as e:
            print(f"Errore caricando {file}: {e}")  # Debug in caso di errore

    print(f"Dataset finale: {len(dataset)} immagini su {len(valid_classes)} classi valide")
    return dataset

def main():
    dataset = retrieve_AgeDB_dataset("data/AgeDB.zip", "data/AgeDB")
    print(f"Numero totale di campioni nel dataset filtrato: {len(dataset)}")

if __name__ == '__main__':
    main()
