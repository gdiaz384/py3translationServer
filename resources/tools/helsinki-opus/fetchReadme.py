"""
This downloads the readme.md files for models from huggingface.co/Helsinki-NLP. All of them.

Docs:
https://huggingface.co/docs/huggingface_hub/en/guides/download
https://huggingface.co/docs/huggingface_hub/main/en/guides/search
https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_api#huggingface_hub.HfApi.list_models
https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py#L2033
https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_files
https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache
"""
__version__ = 2025.06.26

readmeFilename = 'README.md'
collectionsFile = 'opusmtMultilingualTCBibleCollection.txt'
masterList = 'masterList.txt'


import sys
import pathlib

if os.getenv('HF_HUB_DISABLE_TELEMETRY') != '1':
    os.environ[ 'HF_HUB_DISABLE_TELEMETRY' ] = '1'
import huggingface_hub


api = huggingface_hub.HfApi()

#https://huggingface.co/collections/Helsinki-NLP/opus-mt-multilingual-tcbible-670570b993cf2ffd1749334b
#query = api.list_models( author='Helsinki-NLP', sort='createdAt' )
collection = api.get_collection( 'Helsinki-NLP/opus-mt-multilingual-tcbible-670570b993cf2ffd1749334b' )
opusmtMultilingualTCBibleCollection = [ ]
for model in collection.items:
    if model.item_type == 'model':
        opusmtMultilingualTCBibleCollection.append( model.item_id )
opusmtMultilingualTCBibleCollection = sorted( opusmtMultilingualTCBibleCollection )

#for repositoryID in opusmtMultilingualTCBibleCollection:
#    print( repositoryID )

if pathlib.Path( collectionsFile ).is_file() == False:
    with open( collectionsFile, 'w', encoding='utf-8' ) as outfile:
        for repositoryID in opusmtMultilingualTCBibleCollection:
            outfile.write( repositoryID + '\n' )
    print( 'Wrote:', collectionsFile)


#sys.exit( 0 )


repositoryIDs = [ ]
query = api.list_models( author='Helsinki-NLP', sort='createdAt' )
for model in query:
    repositoryIDs.append( model.id )
    #print(model.id)
repositoryIDs = sorted( repositoryIDs )

if pathlib.Path( masterList ).is_file() == False:
    with open( masterList, 'w', encoding='utf-8' ) as outfile:
        for repositoryID in repositoryIDs:
            outfile.write( repositoryID + '\n' )
    print( 'Wrote:', masterList )

for repositoryID in repositoryIDs:
    filename = huggingface_hub.try_to_load_from_cache( repo_id=repositoryID, filename=readmeFilename )
    if filename == None:
        print( 'Downloading', repositoryID, readmeFilename )
        try:
            filename = huggingface_hub.hf_hub_download( repo_id=repositoryID, filename=readmeFilename )
        except:
            print( 'Error downloading:', repositoryID, readmeFilename )

