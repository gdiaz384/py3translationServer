"""
Docs:
https://huggingface.co/docs/huggingface_hub/en/guides/download
https://huggingface.co/docs/huggingface_hub/main/en/guides/search
https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_api#huggingface_hub.HfApi.list_models
https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py#L2033
https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_files
https://huggingface.co/docs/huggingface_hub/main/en/package_reference/cache#huggingface_hub.try_to_load_from_cache
https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache#chunk-based-caching-xet
"""
__version__ = '2025.07.10'


knownBinNames = [ 'model.safetensors', 'pytorch_model.bin', 'tf_model.h5', 'model.bin', 'big.pretrain.pt', 'rust_model.ot', 'flax_model.msgpack' ]
excludeTheseFiles = [ 'benchmark_translations.zip' ] # '.gitattributes',  # gitattributes contains a lot of important metadata for where and how huggingface_hub stores files (LFS or XET)
# https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache#chunk-based-caching-xet


import sys
import os
import pathlib

if os.getenv( 'HF_HUB_DISABLE_TELEMETRY' ) != '1':
    os.environ[ 'HF_HUB_DISABLE_TELEMETRY' ] = '1'
import huggingface_hub


#Helsinki-NLP/opus-mt-en-it
if len( sys.argv ) == 2:
    repository = sys.argv[ 1 ]
else:
    sys.exit( 0 )
    repository='Helsinki-NLP/opus-mt-tc-bible-big-afa-en'
    repository2='https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-afa-en'

#reference = huggingface_hub.list_repo_refs( repo_id=repository )
#print( 'Downloading', repository )
#for i in reference.branches:
#    print(i.name)
#print( 'Using:', reference.branches[0].name )
#commit=reference.branches[ 0 ].target_commit

url = 'https://huggingface.co/' + repository
#https://huggingface.co/Helsinki-NLP/opus-mt-en-ha
if repository.startswith( 'http' ):
    url = repository
    temp = repository.rsplit( '/', maxsplit=1 )
    repositoryName = temp[ 1 ]
    uploader = temp[ 0 ].rsplit( '/', maxsplit=1 )[ 1 ]
    repository = uploader + '/' + repositoryName

print( 'Searching cache for', repository )
if huggingface_hub.try_to_load_from_cache( repo_id=repository, filename='.gitattributes' ) != None:
    downloadedTo = str( pathlib.Path( huggingface_hub.try_to_load_from_cache( repo_id=repository, filename='.gitattributes' ) ).parent )
    print( 'Downloaded to', downloadedTo )
    sys.exit( 0 )

print( 'Downloading', repository, 'from', url )

repositoryFileList = huggingface_hub.list_repo_files( repo_id=repository )
# Returns as: ['.gitattributes', 'README.md', 'benchmark_results.txt', 'benchmark_translations.zip', 'config.json', 'generation_config.json', 'model.safetensors', 'pytorch_model.bin', 'source.spm', 'special_tokens_map.json', 'target.spm', 'tokenizer_config.json', 'vocab.json']

count = 0
for filename in knownBinNames:
    if filename in repositoryFileList:
        count += 1

requestList = [ ]
# Simple case
if ( count == 0 ) or ( count == 1 ):
    for filename in repositoryFileList:
        if ( filename in excludeTheseFiles ) == False:
            requestList.append( filename )
#elif count > 1:
else:
    includeNextModelBin = True
    for filename in repositoryFileList:
        if ( ( filename in excludeTheseFiles ) == False ) and ( ( filename in knownBinNames ) == False ):
            requestList.append( filename )
        elif includeNextModelBin == True:
            if ( filename in knownBinNames ) == True:
                requestList.append( filename )
                includeNextModelBin = False

print( 'full repositoryFileList:', repositoryFileList )
print( 'trimmed requestList:', requestList )
#str( pathlib.Path( huggingface_hub.try_to_load_from_cache( repo_id=repository, filename='model.safetensors' ) ).parent )
#str( pathlib.Path( huggingface_hub.try_to_load_from_cache( repo_id=repository, filename='.gitattributes' ) ).parent )
#print('try_to_load_from_cache',str(huggingface_hub.try_to_load_from_cache( repo_id=repository, filename='.gitattributes' )) )

huggingface_hub.snapshot_download( repo_id=repository, allow_patterns=requestList )  # ignore_patterns= )

downloadedTo = str( pathlib.Path( huggingface_hub.try_to_load_from_cache( repo_id=repository, filename='.gitattributes' ) ).parent )
print( 'Downloaded to', downloadedTo )
