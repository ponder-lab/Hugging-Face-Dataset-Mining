from huggingface_hub import HfApi
import json

#retrieves all hugging face datasets that are tabular and in csv format. 
#Successfully retrieved 8481 datasets

api = HfApi()

datasets_gen = api.list_datasets(limit=None, full=True)
datasets = list(datasets_gen)

dataset_ids = []

for dataset in datasets:
    tags = dataset.tags or []
    
    is_tabular = 'modality:tabular' in tags
    is_csv = 'format:csv' in tags
    
    if is_tabular and is_csv:
        dataset_ids.append(dataset.id)

with open("filtered_datasets.json", "w") as f:
    json.dump(dataset_ids, f, indent=2)

