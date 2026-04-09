#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Description:
A plugin library to add support for transformers.opus-mt to py3translationServer.

Usage:
import importlib
opusmt = importlib.import_module( 'resources.engines.transformers.opusmt.opus-mt' )
or
import sys
sys.modules[ 'opus-mt' ] = __import__( 'resources.engines.transformers.opusmt.opus-mt' )
opusmt = sys.modules[ 'opus-mt' ]
or
import sys
import importlib
#sys.path.append( str( pathlib.Path( __file__ ).parent.parent ) ) # This adds one directory up from os.getcwd() to sys.path assuming the current directory is where the opus-mt.py file is.
sys.path.append( str( pathlib.Path( 'C:/Users/Public/Downloads/apps/server/engines/opusmt/opus-mt.py' ).parent ) )
opusmt = importlib.import_module( 'opus-mt' )
or
Note that if the python filename contains ., like folder/my.module.name.py, that is not importable directly either with import, __import__, or importlib.import_module() because that is a reserved character. Copy of the file, ideally to the same folder, replace the . with _, and then try importing it again, like importlib.importmodule( 'folder.my_module_name' ).
And finally:
translator = opusmt.Translator( modelName='opus-mt-it-en', modelPath=C:/my/model/path/to/model.bin, modelSettings={ } )
print( translator.translate( [ 'a', 'list', 'of', 'strings' ] )

Requirements:
Python 3.8+
python -m pip install transformers hugginface_hub sacremoses
pytorch, https://pytorch.org/get-started/previous-versions/ 
For CUDA 11, use PyTorch <= 2.3.x
For CUDA 12, use PyTorch >= 2.5.x
Tested working on transformers==4.44.2 huggingface_hub==0.33.0 pytorch==2.3.1+cpu pytorch==2.3.0+cu118 sacremoses==0.1.1

Transformers and huggingface_hub, source code and documentation:
https://github.com/huggingface/transformers
https://huggingface.co/docs/transformers/index
https://huggingface.co/docs/transformers/main/en/installation#offline-mode
https://huggingface.co/docs/huggingface_hub/main/en/guides/manage-cache
https://huggingface.co/docs/huggingface_hub/main/en/guides/manage-cache#clean-your-cache
https://huggingface.co/docs/huggingface_hub/main/en/package_reference/cache
Examples: https://github.com/huggingface/transformers/blob/main/awesome-transformers.md

Helsinki-NLP:
https://huggingface.co/collections/Helsinki-NLP
Opus-MT benchmarks:
https://opus.nlpl.eu/dashboard/index.php

Transformers model cache location:
Windows: %HOMEPATH%\.cache\huggingface\hub
Linux: $HOME/.cache/huggingface/hub  or  $PWD/.cache/huggingface/hub
for repo in huggingface_hub.scan_cache_dir().repos:
    print( repo.repo_type, repo.repo_id, repo.size_on_disk_str, repo.repo_path)
#or
for repo in huggingface_hub.scan_cache_dir().repos:
    print( repo.repo_type, repo.repo_id, repo.size_on_disk_str, '', end='')
    for i in repo.revisions:
        print( str( i.snapshot_path ) )

Copyright (c) gdiaz384; License: See main program.
"""
__version__ = '2025.06.25'


# List models.
# https://huggingface.co/collections/Helsinki-NLP/opus-mt-multilingual-tcbible-670570b993cf2ffd1749334b
# This gets overwritten later with data obtained from opus-mt.csv, but here is what the models { } data structure looks like.
models = { }
models[ 'defaultModel' ] = 'opus-mt-tc-bible-big-aav-fra_ita_por_spa' #Required.
opus_mt_tc_bible_big_aav_fra_ita_por_spa = { }
opus_mt_tc_bible_big_aav_fra_ita_por_spa[ 'modelPath' ] = None
opus_mt_tc_bible_big_aav_fra_ita_por_spa[ 'sourceLanguages' ] = ( 'bru', 'cmo', 'hoc', 'jun', 'kha', 'khm', 'kxm', 'vie', 'wbm' )
opus_mt_tc_bible_big_aav_fra_ita_por_spa[ 'targetLanguages' ] = ( 'fra', 'ita', 'por', 'spa' )
opus_mt_tc_bible_big_aav_fra_ita_por_spa[ 'modelURL' ] = [ 'https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-aav-fra_ita_por_spa' ]
opus_mt_tc_bible_big_aav_fra_ita_por_spa[ 'description' ] = 'Neural machine translation model for translating from Austro-Asiatic languages (aav) to unknown (fra+ita+por+spa).'
opus_mt_tc_bible_big_aav_fra_ita_por_spa[ 'prependTargetLanguage' ] = True
models[ 'opus-mt-tc-bible-big-aav-fra_ita_por_spa' ] = opus_mt_tc_bible_big_aav_fra_ita_por_spa
opus_mt_tc_bible_big_afa_en = { }
opus_mt_tc_bible_big_afa_en[ 'sourceLanguages' ] = ( 'aar', 'acm', 'afb', 'amh', 'apc', 'ara', 'arc', 'arq', 'arz', 'bcw', 'byn', 'cop', 'daa', 'dsh', 'gde', 'gnd', 'hau', 'hbo', 'heb', 'hig', 'irk', 'jpa', 'kab', 'ker', 'kqp', 'ktb', 'kxc', 'lln', 'lme', 'meq', 'mfh', 'mfi', 'mfk', 'mif', 'mlt', 'mpg', 'mqb', 'muy', 'oar', 'orm', 'pbi', 'phn', 'rif', 'sgw', 'shi', 'shy', 'som', 'sur', 'syc', 'syr', 'taq', 'tig', 'tir', 'tmc', 'tmh', 'tmr', 'ttr', 'tzm', 'wal', 'xed', 'zgh' )
opus_mt_tc_bible_big_afa_en[ 'targetLanguages' ] = ( 'eng' )
opus_mt_tc_bible_big_afa_en[ 'modelPath' ] = None
opus_mt_tc_bible_big_afa_en[ 'modelURL' ] = [ 'https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-afa-en' ]
opus_mt_tc_bible_big_afa_en[ 'prependTargetLanguage' ] = False
models[ 'opus-mt-tc-bible-big-afa-en' ] = opus_mt_tc_bible_big_afa_en
sugoilevi = { }
sugoilevi[ 'sourceLanguages' ] = ( 'ja' )
sugoilevi[ 'targetLanguages' ] = ( 'en' )
sugoilevi[ 'modelPath' ] = None
sugoilevi[ 'modelURL' ] = [ 'https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-afa-deu_eng_nld' ]
opus_mt_tc_bible_big_afa_en[ 'prependTargetLanguage' ] = False
models[ 'opus-mt-tc-bible-big-afa-deu_eng_nld' ] = sugoilevi
sugoilevi = { }
sugoilevi[ 'sourceLanguages' ] = ( 'ja' )
sugoilevi[ 'targetLanguages' ] = ( 'en' )
sugoilevi[ 'modelPath' ] = None
sugoilevi[ 'modelURL' ] = [ 'https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-afa-deu_eng_fra_por_spa' ]
opus_mt_tc_bible_big_afa_en[ 'prependTargetLanguage' ] = False
models[ 'opus-mt-tc-bible-big-afa-deu_eng_fra_por_spa' ] = sugoilevi

# Initialize static values.
# Use these language codes:
# https://huggingface.co/collections/Helsinki-NLP/opus-mt-multilingual-tcbible-670570b993cf2ffd1749334b
# -1 is cpu. cpu is aliased to -1. 0-2 are references to different CUDA devices. cuda is aliased to 0. To list the first cuda device, run the following command at a cli. To list the second one, change the 0 to a 1. If the command gives an error, then cuda is not available or there is no cuda device at that index.
# python -c "import torch; print( torch.cuda.get_device_properties( 0 ) )"
devices = ( 'cpu', 'cuda', '-1', '0', '1', '2' )
defaultSourceLanguage = 'vie'
defaultTargetLanguage = 'spa'
#defaultSourceLanguage = 'jpn'
#defaultTargetLanguage = 'eng'
defaultModelBinName = 'model.safetensors'
defaultModelBinNames = [ 'model.safetensors', 'pytorch_model.bin', 'tf_model.h5' ]
defaultCSVEncoding = 'utf-8'
#(None), unix, excel, excel-tab
defaultCSVDialect = None
modelsCsvSuffix = '.csv'
inputErrorHandling = 'strict'
consoleEncoding = 'utf-8'
debug = False

# https://huggingface.co/docs/transformers/model_doc/marian
# https://huggingface.co/docs/transformers/v4.44.2/en/main_classes/pipelines#transformers.TranslationPipeline
default_num_beams = 1
# The maximum number of tokens to generate. Must be defined and < 512.
default_max_length = 256
# -1 means to only use the CPU. A non-negative integer means to use the GPU at that index. Example: 0. Default = None which will list the available CUDA devices and autodetect.
# device (int, optional, defaults to -1) — "Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too."
#device=None


import sys
import os
import pathlib
import csv

# https://huggingface.co/docs/transformers/main/en/installation#offline-mode
# https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhubdisabletelemetry
if os.getenv( 'HF_HUB_OFFLINE' ) != '1':
    os.environ[ 'HF_HUB_OFFLINE' ] = '1'
if os.getenv( 'HF_HUB_DISABLE_TELEMETRY' ) != '1':
    os.environ[ 'HF_HUB_DISABLE_TELEMETRY' ] = '1'

#import ctranslate2
#import sentencepiece
import transformers
#import transformers.MarianMTModel
#import transformers.MarianTokenizer
#import huggingface_hub
#import torch
#import torch_directml

try:
    import resources.commonFunctions as commonFunctions
except ImportError:
    sys.path.append( str( pathlib.Path( __file__ ).resolve().parent.parent.parent ) )
    #print( sys.path )
    try:
        import commonFunctions
    except ImportError:
        sys.path.append( str( pathlib.Path( __file__ ).resolve().parent.parent.parent.parent ) )
        #try:
        import commonFunctions
        #except ImportError:
        pass

#Or to import directly:
#import sys
#import pathlib
#sys.path.append( str( pathlib.Path( 'C:\\resources\\functions.py' ).resolve().parent ) )
#import commonFunctions



def createOpusmtModelsDictionaryFromCSVFile( filename, csvEncoding=defaultCSVEncoding, csvDialect=defaultCSVDialect ):
    tempDB = { }
    with open( filename, 'r', newline='', encoding=csvEncoding, errors=inputErrorHandling ) as myFile:
        if csvDialect == None:
            myCsvHandle = csv.reader( myFile )
        else:
            myCsvHandle = csv.reader( myFile, dialect=csvDialect )
        for counter,listOfStringsRow in enumerate( myCsvHandle ):
            if counter == 0:
                continue
            for i in range( len( listOfStringsRow ) ):
                # Clean up whitespace for entities.
                listOfStringsRow[ i ] = listOfStringsRow[ i ].strip()
                # Fix types.
                if listOfStringsRow[ i ].lower() == 'true':
                    listOfStringsRow[ i ] = True
                elif listOfStringsRow[ i ].lower() == 'false':
                    listOfStringsRow[ i ] = False
                elif ( listOfStringsRow[ i ].lower() == 'none' ) or ( listOfStringsRow[ i ].lower() == '' ):
                    listOfStringsRow[ i ] = None
                # Leave numbers as strings. They should not be processed anyway, so there is no need to mess with them.
            #name, modelPath, sourceLanguages, targetLanguages, prependTargetLanguage, modelURL, description
            #name, modelPath, sourceLanguages, targetLanguages, modelURL, description, prependTargetLanguage
            #name, modelPath, sourceLanguages, targetLanguages, defaultSourceLanguage, defaultTargetLanguage, modelURLs, description, prependTargetLanguage
            if listOfStringsRow[ 6 ] != None:
                if listOfStringsRow[ 6 ].find( ',' ) == -1:
                    modelURLs=[ listOfStringsRow[ 6 ] ]
                else:
                    modelURLs = listOfStringsRow[ 6 ].split( ',' )
                    for counter,url in enumerate( modelURLs ):
                        modelURLs[ counter ] = modelURLs[ counter ].strip()
            tempDB[ listOfStringsRow[ 0 ] ] = {
                                                                        'modelPath' : listOfStringsRow[ 1 ],
                                                                        'sourceLanguages' : listOfStringsRow[ 2 ].split( ' ' ),
                                                                        'targetLanguages' : listOfStringsRow[ 3 ].split( ' ' ),
                                                                        'sourceLanguage' : listOfStringsRow[ 4 ],
                                                                        'targetLanguage' : listOfStringsRow[ 5 ],
                                                                        'modelURLs' : modelURLs,
                                                                        'description' : listOfStringsRow[ 7 ],
                                                                        'prependTargetLanguage' : listOfStringsRow[ 8 ],
                                                                        }
    return tempDB

#databaseFile = 'helsinki-opus/opus-tc-bible-big-models.csv'
#databaseFile = current path + '/' + module_name.csv
databaseFile = str( pathlib.Path( __file__ ).parent ) + '/' + pathlib.Path( __file__ ).stem + modelsCsvSuffix
models = createOpusmtModelsDictionaryFromCSVFile( databaseFile )
models[ 'defaultModel' ] = 'opus-mt-tc-bible-big-afa-en' #Required
#models[ 'defaultModel' ] = 'opus-mt-tc-bible-big-aav-fra_ita_por_spa' #Required


class Translator():
    # 'modelSettings={ }' is equivalent to '**modelSettings' provided the function is passed parameters as keywords instead of a dictionary. Only use modelSettings for optional values or module specific values.
    def __init__( self, modelName=None, modelPath=None, device=None, modelSettings={ } ):
        # Initialize generic variables.
        self.modelName = modelName
        if self.modelName == 'defaultModel':
            self.modelName = models[ 'defaultModel' ]
        assert( ( self.modelName in models ) == True )
        assert( pathlib.Path( modelPath ).exists() == True )
        if pathlib.Path( modelPath ).is_dir() == True:
            self.modelPath = modelPath + '/' + defaultModelBinNames[ 0 ]
        else:
            self.modelPath = modelPath
        assert( pathlib.Path( self.modelPath ).is_file() == True )
        self.modelFolderOnly = str( pathlib.Path( self.modelPath ).parent.resolve() )
        self.modelPath = commonFunctions.fixPath( modelPath, basePath=os.getcwd() ) # This will resolve any symbolic links for the modelPath which means '.cache/huggingface/hub/model/snapshot/commit/model.bin' will get converted to '.cache/huggingface/hub/model/blob/hash', meaning that it can have an entirely different base folder.
        #print( 'modelPath', modelPath )
        #print( 'self.modelPath', self.modelPath )
        #print( self.modelFolderOnly )
        if 'device' != None:
            self.device = device
        else:
            self.device = None
        self.device = commonFunctions.getDevice( self.device )
        if 'hash' in modelSettings:
            self.hash = modelSettings[ 'hash' ]
        else:
            self.hash = commonFunctions.getSHA1HashOfFile( self.modelPath )
        self.lastSourceLanguage = None
        self.lastTargetLanguage = None

        # Initialize model specific variables.
        tempVersion = transformers.__version__.split( '.' ) # These are all still strings.
        self.engineMajorVersion = int( tempVersion[ 0 ] )
        self.engineMinorVersion = int( tempVersion[ 1 ] )
        self.enginePatchVersion = int( tempVersion[ 2 ] )
        if 'num_beams' in modelSettings:
            self.num_beams = modelSettings[ 'num_beams' ]
        else:
            self.num_beams = default_num_beams
        if 'max_length' in modelSettings:
            self.max_length = int( modelSettings[ 'max_length' ] )
        else:
            self.max_length = default_max_length

        #https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-afa-en
        # Initialize the tokenizer.
        self.tokenizer = transformers.MarianTokenizer.from_pretrained( self.modelFolderOnly, clean_up_tokenization_spaces=True )

        # Initialize the model.
        self.translator = transformers.MarianMTModel.from_pretrained( self.modelFolderOnly, num_beams=self.num_beams )
        # Workaround for bug. https://github.com/huggingface/transformers/issues/25139
        # This bug was fixed in transformers 4.45.0, so transformers==4.44.2 still has it.
        if self.engineMajorVersion <= 4:
            if ( ( self.engineMinorVersion <= 44 ) and ( self.engineMajorVersion == 4 ) ) or ( self.engineMajorVersion < 4 ):
                self.translator.generation_config.max_length = self.max_length #511 # Setting this to 512 or leaving it as None will cause a crash prior to 4.45.0. Some of the transformers documentation implies the default value for this is 256.


    def encode( self, rawText, sourceLanguage=None, targetLanguage=None ):
        print( 'prepend', models[ self.modelName ][ 'prependTargetLanguage' ] )
        if models[ self.modelName ][ 'prependTargetLanguage' ] == False:
            tempList = self.tokenizer( rawText, return_tensors='pt', padding=True )
            #print( 'pie' )
        else:
            tempList = [ ]
            for i in rawText:
                tempList.append( '>>' + targetLanguage + '<< '+ i )
            tempList = self.tokenizer( tempList, return_tensors='pt', padding=True )
        return tempList


    def translate( self, rawText, sourceLanguage=None, targetLanguage=None ):
        assert( isinstance( rawText, list) )
        # Update the self.lastSourceLanguage and self.lastTargetLanguage
        if sourceLanguage == None:
            self.lastSourceLanguage = defaultSourceLanguage
        else:
            if ( self.lastSourceLanguage == None ) or ( self.lastSourceLanguage != sourceLanguage ):
                self.lastSourceLanguage = sourceLanguage
        if targetLanguage == None:
            self.lastTargetLanguage = defaultTargetLanguage
        else:
            if ( self.lastTargetLanguage == None ) or ( self.lastTargetLanguage != targetLanguage ):
                self.lastTargetLanguage = targetLanguage

        # Tokenize rawText.
        encodedList = self.encode( rawText, sourceLanguage, targetLanguage )

        #params = {'num_beams' : self.num_beams,}
        #if self.engineMajorVersion >= 4:
            #if self.engineMinorVersion >= 4:
                # return_logits_vocab was added to ctranslate2.Translator().translate_batch() as a parameter starting at version 4.4.0, so only include it for that version or newer.
                #params[ 'return_logits_vocab' ] = self.return_logits_vocab
                #pass

        # ** means unpack the dictionary to key=value pairs suitable for use as function keyword parameters. One * means to unpack the dictionary's keynames and use them as positional parameters, not the value of the keys.
        #translatedList = self.translator.translate_batch( encodedList, **params )
        # This strange ** syntax for the encodedList is Required for batch syntax in transformers. There is no other way to feed the data, which is an instance of <class 'transformers.tokenization_utils_base.BatchEncoding'>, to the model as a batch otherwise. Lists do not work, unpacking lists do not work, converting to numpy does not work, sending multiple dictionaries in a list does not work. The only thing that works is sending a single string obtained from tokenizer(string).input_ids. That does work for exactly one input to produce one output without resorting to **. The ** syntax itself works for batches. This means the encodedList is not a list like normal, but an instance, or several instances?, of the above class.
        # https://huggingface.co/docs/transformers/v4.44.2/en/main_classes/tokenizer#transformers.BatchEncoding
        # This class is derived from a python dictionary and can be used as a dictionary. In addition, this class exposes utility methods to map from word/character space to token space. -docs
        translatedList = self.translator.generate( **encodedList )
        return self.decode( translatedList, sourceLanguage, targetLanguage )


    def decode( self, translatedList, sourceLanguage=None, targetLanguage=None ):
        return self.tokenizer.batch_decode( translatedList, skip_special_tokens=True )
        #decodedList = [ ]
        #for i in translatedList:
        #    decodedList.append( self.tokenizer.decode( i, skip_special_tokens=True ) )
        #return decodedList

