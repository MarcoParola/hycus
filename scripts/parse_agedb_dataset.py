import zipfile
import os
from PIL import Image
from collections import defaultdict

def retrieve_AgeDB_dataset(zip_path, extract_folder):
    # Estrai lo ZIP
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    # Trova la sottocartella se esiste
    subfolders = [f for f in os.listdir(extract_folder) if os.path.isdir(os.path.join(extract_folder, f))]
    if subfolders:
        extract_folder = os.path.join(extract_folder, subfolders[0])  # Prende la prima sottocartella

    # Lista immagini
    file_list = [f for f in os.listdir(extract_folder) if f.endswith(".jpg")]

    # Dizionario nome → ID e conteggio delle immagini per classe
    name_to_id = {}
    class_counts = defaultdict(int)
    print("Sample totali ", len(file_list))

    # Conta le immagini per ogni classe (nome)
    for file in file_list:
        parts = file.split("_")
        if len(parts) < 4:
            continue  # Salta file con nomi non conformi
        nome = parts[1].lower()  # Normalizza il nome
        class_counts[nome] += 1
    print("Classi totali ", len(class_counts))
    # Filtra le classi con meno di 10 campioni
    valid_classes = {name for name, count in class_counts.items() if count >= 10}

    dataset = []

    # Creazione dataset con solo le classi valide
    for file in file_list:
        parts = file.split("_")
        if len(parts) < 4:
            continue

        id_foto, nome, età, sesso = parts[:4]  
        nome = nome.lower()

        if nome not in valid_classes:
            continue  # Salta le classi con meno di 10 campioni

        if nome not in name_to_id:
            name_to_id[nome] = len(name_to_id)  # Assegna ID univoco

        img_path = os.path.join(extract_folder, file)
        
        try:
            img = Image.open(img_path).convert("RGB")  # Carica immagine PIL
            dataset.append((img, name_to_id[nome]))  # Salva (immagine, ID)
        except Exception as e:
            print(f"Errore caricando {file}: {e}")  # Debug in caso di errore

    print(f"Dataset finale: {len(dataset)} immagini su {len(valid_classes)} classi valide")
    return dataset

def main():
    dataset = retrieve_AgeDB_dataset("data/AgeDB.zip", "data/AgeDB")
    print(f"Numero totale di campioni nel dataset filtrato: {len(dataset)}")

if __name__ == '__main__':
    main()
