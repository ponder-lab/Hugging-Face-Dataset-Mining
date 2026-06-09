import os
import time
import pandas as pd
import re
import requests
import json
from requests.exceptions import JSONDecodeError

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from huggingface_hub import HfApi, login, SpaceHardware
from huggingface_hub.hf_api import ModelInfo, RepoFile
from huggingface_hub import hf_hub_url, get_hf_file_metadata
from huggingface_hub.utils import GatedRepoError
from huggingface_hub.utils import HfHubHTTPError, EntryNotFoundError

api = HfApi()

#Retrieves the commit logs from the previously retrieved dataset 

with open('filtered_datasets.json', 'r') as f:
    filtered_dataset_ids = json.load(f)

print(f"Loaded {len(filtered_dataset_ids)} filtered dataset IDs")


all_datasets_gen = api.list_datasets(full=True, limit=None)
all_datasets = list(all_datasets_gen)

datasets_api_dict = {dataset.id: dataset for dataset in all_datasets}
datasets = [datasets_api_dict.get(dataset_id, dataset_id) for dataset_id in filtered_dataset_ids]


def retrieve_dataset_tags(dataset):

    
    tags = list(dataset.tags or [])
    if hasattr(dataset, 'cardData') and dataset.cardData and 'tags' in dataset.cardData:
        if type(dataset.cardData['tags']) is list:
            try:
                tags = list(set(tags + dataset.cardData['tags']))
            except:
                print(dataset.cardData['tags'])
        else:
            tags = list(set(tags + [dataset.cardData['tags']]))

    tags = [tag for tag in tags if tag is not None]
    return tags

# def find_dataset_size(dataset):
#     """
#     Find the size of datasets used by a given dataset.
#     """    
    
#     dataset_size = 0
#     if dataset is None:
#         return None
#     api_token = os.environ["HF_TOKEN"]  # Replace with your token

#     try:
#         dataset_size += api.dataset_info(dataset, token=api_token).cardData["dataset_info"]["dataset_size"]
#     except:
#         pass

#     return dataset_size

def api_calls_parameters(dataset):
    """
    Get size, datasets size, and creation date from API calls.
    
    Args:
        dataset: The dataset object.
    
    Returns:
        A tuple containing files, commits, size, and created_at.
    """
    
    commits = created_at = None
    api_token = os.environ["HF_TOKEN"]  # Your token
    
    try:
        # IMPORTANT: Add repo_type="dataset" for dataset repositories
        files = api.list_repo_files(repo_id=dataset.id, token=api_token, repo_type="dataset")
    except GatedRepoError:
        print(f'Need authorization to retrieve files and commits from {dataset.id}')
        files = 'needs authorization'
    except Exception as e:
        print(f'Unexpected error on retrieving "files" for {dataset.id}:', str(e))
        files = None
        
    try:
        # IMPORTANT: Add repo_type="dataset" for dataset repositories
        commits = api.list_repo_commits(repo_id=dataset.id, token=api_token, repo_type="dataset")
        commits = [{**commit.__dict__, "created_at": commit.created_at.strftime('%Y-%m-%dT%H:%M:%S.%fZ')[:-3]+'Z'} for commit in commits]
        created_at = commits[-1]['created_at']
        commits = json.dumps(commits)
    except GatedRepoError:
        commits = 'not authorized'
    except Exception as e:
        print(f'Unexpected error on retrieving "commits" for {dataset.id}:', str(e))
        commits = None 

    return files, commits, created_at


# def find_dataset_size(dataset_id):
    """
    Find the size of datasets used by a given model.
    
    Args:
        dataset_id: The dataset ID string.
    
    Returns:
        The total size of the datasets or None if not found.
    """    
    
    dataset_size = 0
    if dataset_id is None:
        return None
    api_token = os.environ["HF_TOKEN"]

    try:
        # IMPORTANT: Add repo_type="dataset" for dataset repositories
        dataset_info = api.dataset_info(dataset_id, token=api_token)
        if hasattr(dataset_info, 'cardData') and dataset_info.cardData and "dataset_info" in dataset_info.cardData:
            dataset_size += dataset_info.cardData["dataset_info"]["dataset_size"]
    except Exception as e:
        print(f'Error getting size for {dataset_id}: {str(e)}')
        pass

    return dataset_size if dataset_size > 0 else None


# Alternative: More robust version with better error handling
def api_calls_parameters_robust(dataset):
    """
    Robust version with better error handling and repo_type specification.
    """
    
    commits = created_at = None
    files = None
    api_token = os.environ["HF_TOKEN"]
    
    dataset_id = dataset.id if hasattr(dataset, 'id') else str(dataset)
    
    # # Get files
    # try:
    #     files = api.list_repo_files(
    #         repo_id=dataset_id, 
    #         token=api_token, 
    #         repo_type="dataset"
    #     )
    # except GatedRepoError:
    #     files = 'needs authorization'
    # except Exception as e:
    #     if "404" in str(e):
    #         files = 'repository not found'
    #     else:
    #         files = None
    #         print(f'Unexpected error on retrieving "files" for {dataset_id}: {str(e)}')
        
    # Get commits
    try:
        commits_list = api.list_repo_commits(
            repo_id=dataset_id, 
            token=api_token, 
            repo_type="dataset"
        )
        
        if commits_list:
            commits = []
            for commit in commits_list:
                commit_dict = {**commit.__dict__}
                if hasattr(commit, 'created_at') and commit.created_at:
                    commit_dict["created_at"] = commit.created_at.strftime('%Y-%m-%dT%H:%M:%S.%fZ')[:-3]+'Z'
                commits.append(commit_dict)
            
            created_at = commits[-1]['created_at'] if commits else None
            commits = json.dumps(commits)
        
    except GatedRepoError:
        commits = 'not authorized'
    except Exception as e:
        if "404" in str(e):
            commits = 'repository not found'
        else:
            commits = None
            print(f'Unexpected error on retrieving "commits" for {dataset_id}: {str(e)}')

    return commits, created_at


# Updated process_dataset function
def process_dataset(dataset):
    """
    Process a dataset and extract relevant information.

    Args:
        dataset: A tuple containing the dataset object.

    Returns:
        A dictionary containing the processed dataset information.
    """
    
    if dataset[0] % 100 == 0:
        print(dataset[0])

    dataset = dataset[1]
    
    try:
        tags = retrieve_dataset_tags(dataset)
        commits, created_at = api_calls_parameters_robust(dataset)
        
        # Get dataset size
        # size = find_dataset_size(dataset.id)

        return {
            'datasetId': dataset.id,
            'tags': tags,
            'downloads': dataset.downloads,
            'likes': dataset.likes,
            'lastModified': dataset.lastModified,
            'created_at': created_at,
            'commits': commits,
        }
    except Exception as e:
        if isinstance(dataset, str):
            print(dataset, 'is not available anymore')
        else:
            print(f'{dataset.id} could not be processed: ', str(e))
        return None
    
# Prepare datasets for processing
datasets = [(idx, dataset) for idx, dataset in enumerate(datasets)]

print(f"Processing {len(datasets)} datasets...")

start = time.time()

num_threads = 8  # adjust threads

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    datasets_information = list(executor.map(process_dataset, datasets))
    
datasets_information = [dataset for dataset in datasets_information if dataset is not None]
df = pd.DataFrame(datasets_information)
end = time.time()
print(f"Processing took {end - start:.2f} seconds")

df.to_csv('FilteredHFDatasets.csv')
print(f"Saved {len(df)} datasets to FilteredHFDatasets.csv")