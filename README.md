1. Run 1getHFdatasets.py to extract datasets from Hugging Face. Filters such as modularity and file format can be adjusted by editing, adding, or removing lines such as those below.
   is_tabular = 'modality:tabular' in tags
   is_csv = 'format:csv' in tags
   Saves the datasets in "filtered_datasets.json" 
2. Use 2getHFcommits.py to extract more information, including the commit logs, from all the datasets included in "filtered_datasets.json"
   The script extracts datasetId, tags, downloads, likes, lastModified, created_at, commits; saves this information in "FilteredHFDatasets.csv"
3. Run 'python 3HFcommitFormatting.py FilteredHFDatasets.csv outputFilename.csv'
   Formats all previously extracted commits into separate rows; includes DatasetID, CommitId, Authors, Date, Log message, message
