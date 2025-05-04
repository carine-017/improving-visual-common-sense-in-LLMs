import datasets
import os
import requests
from datasets import load_dataset
from typing import Optional, Callable
from pathlib import Path
from tqdm import tqdm

class LaionDatasetLoader:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        
        # Create paths relative to current file location
        self.base_dir = Path(__file__).parent
        self.image_dir = self.base_dir / "data_laion"
        self.dataset_dir = self.base_dir / "datasets"
        
        # Create directories if they don't exist
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_name = "laion/220k-GPT4Vision-captions-from-LIVIS"

    def _download_image(self, url: str) -> Optional[str]:
        """Télécharge une image depuis son URL"""
        try:
            # Crée un nom de fichier unique basé sur l'URL
            filename = str(hash(url)) + '.jpg'
            save_path = self.image_dir / filename
            
            # Si l'image existe déjà, pas besoin de la retélécharger
            if save_path.exists():
                return str(save_path)
            
            # Télécharge l'image
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Sauvegarde l'image
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            return str(save_path)
        except Exception as e:
            print(f"Erreur lors du téléchargement de {url}: {e}")
            return None

    def _create_tokenizer_function(self, caption_column: str) -> Callable:
        def tokenize_examples(examples):
            captions = examples[caption_column]
            
            # Télécharge les images et garde trace des chemins valides
            image_paths = []
            valid_indices = []
            
            for idx, url in enumerate(tqdm(examples["url"], desc="Téléchargement des images")):
                path = self._download_image(url)
                if path:
                    image_paths.append(path)
                    valid_indices.append(idx)
            
            # Ne garde que les captions correspondant aux images téléchargées
            valid_captions = [captions[i] for i in valid_indices]
            
            # Tokenize les captions valides
            text_inputs = self.tokenizer(
                valid_captions, 
                max_length=self.args.max_seq_length,
                padding="max_length",
                truncation=True
            )

            return {
                "input_ids": text_inputs['input_ids'],
                "attention_mask": text_inputs["attention_mask"],
                "image_path": image_paths
            }

        return tokenize_examples

    def _process_dataset(self, dataset, caption_column: str) -> datasets.Dataset:
        tokenize_func = self._create_tokenizer_function(caption_column)
        
        if self.args.max_train_samples:
            dataset = dataset.select(
                range(min(len(dataset), self.args.max_train_samples))
            )

        return dataset.map(
            function=tokenize_func,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=1,  # Important : On met num_proc à 1 pour le téléchargement séquentiel
            load_from_cache_file=not self.args.overwrite_cache,
            desc="Traitement du dataset"
        )

    def load(self, short: bool = False) -> datasets.Dataset:
        dataset_path = self.dataset_dir / f"laion_220_downloaded_{self.args.model_name_or_path}.hf"
        caption_column = 'short_caption' if short else 'caption'

        try:
            return datasets.load_from_disk(str(dataset_path))
        except Exception:
            dataset = load_dataset(self.dataset_name)
            processed_dataset = self._process_dataset(dataset["train"], caption_column)
            processed_dataset.save_to_disk(str(dataset_path))
            return processed_dataset

def load_laion_220(args, tokenizer, short=False):
    loader = LaionDatasetLoader(args, tokenizer)
    return loader.load(short)

