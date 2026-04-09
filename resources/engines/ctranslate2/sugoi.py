#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Description:
A plugin library to add support for ctranslate2.sugoi to py3translationServer.

Usage:
import resources.engines.ctranslate2.sugoi as engine
translator = engine.Translator()
or
sugoiModule = importlib.import_module( resources.engines.ctranslate2.sugoi )
translator = sugoiModule.Translator()
or
sys.path.append( str( pathlib.Path( __file__ ).parent ) )
import sugoi as engine
translator = engine.Translator( name='sugoiv4', modelPath=my/path/to/model.bin, modelSettings={ } )
print( translator.translate( [ 'a', 'list', 'of', 'strings' ] )

Requirements:
python -m pip install ctranslate2 sentencepiece
CUDA also requires pytorch. https://pytorch.org/get-started/previous-versions/ 
For CUDA 11, use ctranslate2 <= 3.x 
For CUDA 12, use ctranslate2 >= 4.x
Tested working on ctranslate2 == 3.24.0 CUDA and 4.6.0 CPU

CTranslate2 source code and documentation:
https://github.com/OpenNMT/CTranslate2
https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html

Copyright (c) gdiaz384; License: See main program.
"""
__version__ = '2025.05.28'


# List models.
models = { }
models[ 'defaultModel' ] = 'sugoi-v4' #Required.
sugoiv4 = { }
sugoiv4[ 'sourceLanguages' ] = ( 'ja', )
sugoiv4[ 'targetLanguages' ] = ( 'en', )
sugoiv4[ 'sourceLanguage' ] = 'ja'
sugoiv4[ 'targetLanguage' ] = 'en'
sugoiv4[ 'modelPath' ] = None
sugoiv4[ 'modelURLs' ] = ( 'https://sugoitoolkit.com', 'https://www.patreon.com/mingshiba' )
sugoiv4[ 'description' ] = 'NMT model for translating Japanese to English by MingShiba.'
models[ 'sugoi-v4' ] = sugoiv4
sugoilevi = { }
sugoilevi[ 'sourceLanguages' ] = ( 'ja', )
sugoilevi[ 'targetLanguages' ] = ( 'en', )
sugoilevi[ 'sourceLanguage' ] = 'ja'
sugoilevi[ 'targetLanguage' ] = 'en'
sugoilevi[ 'modelPath' ] = None
sugoilevi[ 'modelURLs' ] = ( 'https://sugoitoolkit.com', 'https://www.patreon.com/mingshiba' )
sugoilevi[ 'description' ] = 'NMT model for translating Japanese to English by MingShiba. Use the sugoi-v4 model instead of this one.'
models[ 'sugoi-levi' ] = sugoilevi


# Initialize static module values.
# Use two letter language codes: www.loc.gov/standards/iso639-2/php/code_list.php
devices = ( 'cpu', 'cuda' )
defaultSourceLanguage = 'ja'
defaultTargetLanguage = 'en'
#defaultModelBinName = 'big.pretrain.pt'
defaultModelBinName = 'model.bin'
#defaultModelBinName = 'pytorch_model.bin'
#defaultModelBinName = 'tf_model.h5'
defaultModelBinNames = [ 'model.bin' ]
consoleEncoding = 'utf-8'
debug = False

# These are internal variable names for fairseq and CTranslate2, so they use a slightly different variable naming scheme.
# Fairseq documentation and source code:
# https://fairseq.readthedocs.io/en/latest/models.html#fairseq.models.transformer.TransformerModel
# https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_base.py
# https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_legacy.py
# https://fairseq.readthedocs.io/en/latest/_modules/fairseq/models/fairseq_model.html#BaseFairseqModel.from_pretrained
# https://fairseq.readthedocs.io/en/latest/_modules/fairseq/tasks/translation.html?highlight=source_lang
# https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-interactive

# OpenNMT refers to the text preprocessing as as the tokenizer type depending upon model/model format used: byte_bpe, bytes, characters, fastbpe, gpt2, bert, hf_byte_bpe, sentencepiece, subword_nmt
# fairseq uses a different UI: moses, nltk, space.
# This default_bpe uses the value options defined by fairseq.
#default_bpe = 'sentencepiece'
# Example sentence pieces: https://huggingface.co/JustFrederik
# sentencePieceModelFolders is where to search for the sentence piece models. The paths in sentencePieceModelFolders are relative to modelPath, which must be specified at runtime.
# If no sourceLanguageSpm is specified, then use the values in sentencePieceModelFolder together with defaultSourceLanguage to compute a value for sourceLanguageSpm and check if it exists as a file. If it exists, use it. Example:
#'spm/spm.ja.nopretok.model'
# If no targetLanguageSpm is specified, then use the values in sentencePieceModelFolder together with defaultTargetLanguage to compute a value for targetLanguageSpm and check if it exists as a file. If it exists, use it. Example:
#'../spmModel/spm.en.nopretok.model'
sentencePieceModelFolder = [ 'spm', 'spmModel', 'spmModels' ]
sentencePieceModelPrefix = 'spm.'
sentencePieceModelPostfix = '.nopretok.model'

# CTranslate2 documentation:
# https://opennmt.net/CTranslate2/quickstart.html
# https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html
# https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html#ctranslate2.Translator.translate_batch
# Number of OpenMP CPU threads per translator (0 to use a default value). if the psutil library is available, then this will be updated dynamically. Integer, Default=(0)
default_intra_threads = 0
# Maximum number of parallel translations. Higher values increase video memory usage. Seems to have no or little effect on CPU loads and processing time. Integer, Default = (1)
# https://opennmt.net/CTranslate2/parallel.html#data-parallelism
default_inter_threads = 8
# Optional batch of target prefix tokens. Strings generated by a tokenizer put into a list.
default_target_prefix  = None
# Run the translation asynchronously. True, (False)
default_asynchronous = False
# https://fairseq.readthedocs.io/en/latest/_modules/fairseq/tasks/fairseq_task.html?highlight=beam_size
# beam_size is the number of tokens generated by the model. The best one will be chosen as the return value. Directly affects quality. This is the main speed vs quality setting.
# Default=2. Changed to 5 as per default setting in fairseq source code. 
# https://opennmt.net/CTranslate2/guides/fairseq.html#beam-search-equivalence
# Set beam size (1 for greedy search). Best performance is 1. Integer (2)
default_beam_size = 5
# The decoding will continue until beam_size*patience hypotheses are finished.
default_patience = 1.0
# Number of results to return. Integer (1)
default_num_hypotheses = 1
# Exponential penalty applied to the length during beam search. Float (1.0)
default_length_penalty = 1.0
# Coverage penalty weight applied during beam search. Float (0.0)
default_coverage_penalty = 0.0
# Penalty applied to the score of previously generated tokens (set > 1 to penalize). Float (0.0)
default_repetition_penalty = 1.0
# Prevent repetitions of ngrams with this size (set 0 to disable). Integer, Default=0
default_no_repeat_ngram_size = 3
# Disable the generation of the unknown token. True, (False)
default_disable_unk = True
# Disable the generation of some sequences of tokens. A list of strings.
default_suppress_sequences = None
# Stop the decoding on one of these tokens (defaults to the model EOS token). A list of strings or integers.
default_end_token = None
# Include the end token in the results. True, (False)
default_return_end_token = False
# Bias translations towards given prefix. Float (0.0)
default_prefix_bias_beta = 0.0
# Truncate inputs after this many tokens (set 0 to disable). Int (1024)
default_max_input_length = 1024
# Maximum prediction length. Integer (256)
default_max_decoding_length = 256
# Minimum prediction length. Integer (1)
default_min_decoding_length = 1
# Setting this to True corrupts the output, so leave as False until correct vmap can be built. True, (False)
default_use_vmap = False
# Include the scores in the output. True, (False)
default_return_scores = False
# Include the log probs of each token in the output. True, (False), only valid for ctranslate2 >= 4.4.0
default_return_logits_vocab = False
# Include the attention vectors in the output. True, (False)
default_return_attention = False
# Return alternatives at the first unconstrained decoding position. True, (False)
default_return_alternatives = False
# Minimum initial probability to expand an alternative. Float (0.0)
default_min_alternative_expansion_prob = 0.0
# Randomly sample predictions from the top K candidates. Integer (1)
default_sampling_topk = 1
# Keep the most probable tokens whose cumulative probability exceeds this value. Float (1.0)
default_sampling_topp = 1.0
# Sampling temperature to generate more random samples. Float (1.0)
default_sampling_temperature = 1.0
# Replace unknown target tokens by the source token with the highest attention. True, (False)
default_replace_unknowns = False


import sys
import os
import pathlib
import ctranslate2
import sentencepiece
#import torch
#import torch_directml
try:
    import psutil
    psutilAvailable = True
except:
    psutilAvailable = False
try:
    import resources.commonFunctions as commonFunctions
except ImportError:
    sys.path.append( str( pathlib.Path( __file__ ).resolve().parent.parent.parent ) )
    #print( sys.path )
    import commonFunctions
#Or to import directly:
#import sys
#import pathlib
#sys.path.append( str( pathlib.Path( 'C:\\resources\\functions.py' ).resolve().parent ) )
#import commonFunctions


# This gets the ideal number of CPU threads for ctranslate2, so even though it does not reference any object methods or variables besides having a default number of threads, it is still closely associated with the ctranslate2 engine. getCtranslate2IntraThreads() might be a more descriptive name.
def getCpuThreads( intra_threads=default_intra_threads ):
    # Debug code.
    #psutilAvailable=False

    # For best processing time with CTranslate2 + CPU, CPU threads should be the same as the number of physical cores for CPU loads (not logical cores). CPU theads does not matter much when using GPU.
    #If the user specified a number of intra_threads, as --cpuThreads, then just use that instead.
    if ( intra_threads != None ) and ( intra_threads != 0 ):
        return intra_threads

    if psutilAvailable == True:
        #Always gives logical cores. Incorrect.
        #intra_threads = os.cpu_count()
        #Gives physical cores. Correct.
        intra_threads = psutil.cpu_count( logical=False )

        # Setting intra_threads=psutil.cpu_count(logical=False) always gives the wrong value for Bulldozer family FX series processors (2 Module - 4 thread ; 3 Module - 6 thread; 4 Module - 8 thread). Bulldozer FX series should use logical cores, not module count, because every logical core has some dedicated hardware to process the thread, unlike SMT.
        # https://en.wikipedia.org/wiki/List_of_AMD_FX_processors
        # Bandaid for Bulldozer FX systems on Windows. This band-aid fix is currently only available on Windows.
        # This will likely hurt performance for users that have non-Bulldozer AMD FX systems. No modern AMD FX processors currently exist, so this is more of a concern for the future.
        if sys.platform == 'win32':
            try:
                import win32com.client
                if ( str(win32com.client.GetObject('winmgmts:root\cimv2').ExecQuery('Select * from Win32_Processor')[0].Name).strip()[:6].lower() == 'amd fx' ):
                    intra_threads = os.cpu_count()
            except:
                pass
        # Fix for BSD systems. See:
        # https://psutil.readthedocs.io/en/latest/#psutil.cpu_count
        if intra_threads == None:
            intra_threads = default_intra_threads
    elif psutilAvailable == False:
        intra_threads = default_intra_threads
    # Probably pointless, but just in case.
    try:
        assert( isinstance( intra_threads, int ) )
    except:
        print( 'Warning: Could not set CPU threads for CTranslate2 correctly.' )
        intra_threads=0

    return intra_threads


class Translator():
    # 'modelSettings={ }' is equivalent to '**modelSettings' provided the function is passed parameters as keywords instead of a dictionary.
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
        if device != None:
            self.device = None
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
        # The ctranslate2.Translator().translate_batch( 'return_logits_vocab' ) parameter was added in ctranslate2==4.4.0, so check the versions to ensure translate_batch() is getting called correctly.
        tempVersion = ctranslate2.__version__.split( '.' ) # These are all still strings.
        for counter,i in enumerate( tempVersion ):
            tempVersion[ counter ] =int( i )
        self.engineMajorVersion = tempVersion[ 0 ]
        self.engineMinorVersion = tempVersion[ 1 ]
        self.enginePatchVersion = tempVersion[ 2 ]
        if 'intra_threads' in modelSettings:
            self.intra_threads = modelSettings[ 'intra_threads' ]
        else:
            self.intra_threads = default_intra_threads
        self.intra_threads = getCpuThreads( self.intra_threads )
        #print( 'self.intra_threads=', self.intra_threads)
        if 'inter_threads' in modelSettings:
            self.inter_threads = modelSettings[ 'inter_threads' ]
        else:
            self.inter_threads = default_inter_threads
        if 'beam_size' in modelSettings:
            self.beam_size = modelSettings[ 'beam_size' ]
        else:
            self.beam_size = default_beam_size
        if 'patience' in modelSettings:
            self.patience = modelSettings[ 'patience' ]
        else:
            self.patience = default_patience
        if self.patience != None:
            self.patience = float( self.patience )
        if 'num_hypotheses' in modelSettings:
            self.num_hypotheses = modelSettings[ 'num_hypotheses' ]
        else:
            self.num_hypotheses = default_num_hypotheses
        if 'length_penalty' in modelSettings:
            self.length_penalty = modelSettings[ 'length_penalty' ]
        else:
            self.length_penalty = default_length_penalty
        if self.length_penalty != None:
            self.length_penalty = float( self.length_penalty )
        if 'coverage_penalty' in modelSettings:
            self.coverage_penalty = modelSettings[ 'coverage_penalty' ]
        else:
            self.coverage_penalty = default_length_penalty
        if self.coverage_penalty != None:
            self.coverage_penalty = float( self.coverage_penalty )
        if 'repetition_penalty' in modelSettings:
            self.repetition_penalty = modelSettings[ 'repetition_penalty' ]
        else:
            self.repetition_penalty = default_repetition_penalty
        if self.repetition_penalty != None:
            self.repetition_penalty = float( self.repetition_penalty )
        if 'no_repeat_ngram_size' in modelSettings:
            self.no_repeat_ngram_size = modelSettings[ 'no_repeat_ngram_size' ]
        else:
            self.no_repeat_ngram_size = default_no_repeat_ngram_size
        if 'disable_unk' in modelSettings:
            self.disable_unk = modelSettings[ 'disable_unk' ]
        else:
            self.disable_unk = default_disable_unk
        if 'suppress_sequences' in modelSettings:
            self.suppress_sequences = modelSettings[ 'suppress_sequences' ]
        else:
            self.suppress_sequences = default_suppress_sequences
        if 'end_token' in modelSettings:
            self.end_token = modelSettings[ 'end_token' ]
        else:
            self.end_token = default_end_token
        # end_token can be a list of strings or integers.
        if self.end_token != None:
            if self.end_token.find( ',' ) == -1:
                elf.end_token = [ self.end_token ]
            else:
                tempList = self.end_token.split( ',' )
                try:
                    int( tempList[ 0 ] )
                    for counter,i in enumerate( tempList ):
                        tempList[ counter ] = int( tempList[ counter ] )
                except:
                    for counter,i in enumerate( tempList ):
                        tempList[ counter ] = tempList[ counter ].strip()
                self.end_token = tempList
        if 'return_end_token' in modelSettings:
            self.return_end_token = modelSettings[ 'return_end_token' ]
        else:
            self.return_end_token = default_return_end_token
        if 'prefix_bias_beta' in modelSettings:
            self.prefix_bias_beta = modelSettings[ 'prefix_bias_beta' ]
        else:
            self.prefix_bias_beta = default_prefix_bias_beta
        if self.prefix_bias_beta != None:
            self.prefix_bias_beta = float( self.prefix_bias_beta ) 
        if 'max_input_length' in modelSettings:
            self.max_input_length = modelSettings[ 'max_input_length' ]
        else:
            self.max_input_length = default_max_input_length
        if 'max_decoding_length' in modelSettings:
            self.max_decoding_length = modelSettings[ 'max_decoding_length' ]
        else:
            self.max_decoding_length = default_max_decoding_length
        if 'min_decoding_length' in modelSettings:
            self.min_decoding_length = modelSettings[ 'min_decoding_length' ]
        else:
            self.min_decoding_length = default_min_decoding_length
        if 'use_vmap' in modelSettings:
            self.use_vmap = modelSettings[ 'use_vmap' ]
        else:
            self.use_vmap = default_use_vmap
        if 'return_scores' in modelSettings:
            self.return_scores = modelSettings[ 'return_scores' ]
        else:
            self.return_scores = default_return_scores
        if 'return_logits_vocab' in modelSettings:
            self.return_logits_vocab = modelSettings[ 'return_logits_vocab' ]
        else:
            self.return_logits_vocab = default_return_logits_vocab
        if 'return_attention' in modelSettings:
            self.return_attention = modelSettings[ 'return_attention' ]
        else:
            self.return_attention = default_return_attention
        if 'return_alternatives' in modelSettings:
            self.return_alternatives = modelSettings[ 'return_alternatives' ]
        else:
            self.return_alternatives = default_return_alternatives
        if 'min_alternative_expansion_prob' in modelSettings:
            self.min_alternative_expansion_prob = modelSettings[ 'min_alternative_expansion_prob' ]
        else:
            self.min_alternative_expansion_prob = default_min_alternative_expansion_prob
        if self.min_alternative_expansion_prob != None:
            self.min_alternative_expansion_prob = float( self.min_alternative_expansion_prob )
        if 'sampling_topk' in modelSettings:
            self.sampling_topk = modelSettings[ 'sampling_topk' ]
        else:
            self.sampling_topk = default_sampling_topk
        if 'sampling_topp' in modelSettings:
            self.sampling_topp = modelSettings[ 'sampling_topp' ]
        else:
            self.sampling_topp = default_sampling_topp
        if self.sampling_topp != None:
            self.sampling_topp = float( self.sampling_topp )
        if 'sampling_temperature' in modelSettings:
            self.sampling_temperature = modelSettings[ 'sampling_temperature' ]
        else:
            self.sampling_temperature = default_sampling_temperature
        if self.sampling_temperature != None:
            self.sampling_temperature = float( self.sampling_temperature )
        if 'replace_unknowns' in modelSettings:
            self.replace_unknowns = modelSettings[ 'replace_unknowns' ]
        else:
            self.replace_unknowns = default_replace_unknowns

        # Initialize the tokenizer.
        if 'sourceLanguageSpm' in modelSettings:
            self.sourceLanguageSpm = commonFunctions.fixPath( modelSettings[ 'sourceLanguageSpm' ], basePath=self.modelFolderOnly )
        else:
            self.sourceLanguageSpm = None
        if 'targetLanguageSpm' in modelSettings:
            self.targetLanguageSpm = commonFunctions.fixPath( modelSettings[ 'targetLanguageSpm' ], basePath=self.modelFolderOnly )
        else:
            self.targetLanguageSpm = None
        if ( self.sourceLanguageSpm == None ) or ( self.targetLanguageSpm == None ):
            fixSpm=True
        # None cannot be fed into pathlib.Path(), so check for that before invoking pathlib.Path().
        elif ( pathlib.Path( self.sourceLanguageSpm ).is_file() != True ) or ( pathlib.Path( self.targetLanguageSpm ).is_file() != True ):
            fixSpm=True
        else:
            fixSpm=False
        if fixSpm == True:
            self.sentencePieceModelFolderDictionary = { }
            for folder in sentencePieceModelFolder:
                self.sentencePieceModelFolderDictionary[ folder ] = None
            if 'sentencePieceModelFolder' in modelSettings:
                if modelSettings[ 'sentencePieceModelFolder' ] != None:
                    if modelSettings[ 'sentencePieceModelFolder' ].find( ',' ) == -1:
                        self.sentencePieceModelFolderDictionary[ modelSettings[ 'sentencePieceModelFolder' ] ] = None
                    else:
                        tempList = modelSettings[ 'sentencePieceModelFolder' ].split( ',' )
                        for folder in tempList:
                            self.sentencePieceModelFolderDictionary[ folder.strip() ] = None
            self.sourceLanguageSpm, self.targetLanguageSpm = self._getSpmPath( self.modelFolderOnly )
        assert( pathlib.Path( self.sourceLanguageSpm ).is_file() == True )
        assert( pathlib.Path( self.targetLanguageSpm ).is_file() == True )
        self.sourceLanguageTokenizer = sentencepiece.SentencePieceProcessor( self.sourceLanguageSpm )
        self.targetLanguageTokenizer = sentencepiece.SentencePieceProcessor( self.targetLanguageSpm )

        # Initialize the model.
        # Workaround for bug.
        if os.getenv('KMP_DUPLICATE_LIB_OK') != 'TRUE':
            os.environ[ 'KMP_DUPLICATE_LIB_OK' ] = 'TRUE'
        # https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html
        self.translator = ctranslate2.Translator(
                                                        self.modelFolderOnly,
                                                        device=self.device,
                                                        intra_threads=self.intra_threads,
                                                        inter_threads=self.inter_threads,)


    def encode( self, rawText, sourceLanguage=None, targetLanguage=None ):
        return self.sourceLanguageTokenizer.encode( rawText, out_type=str );


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

        # https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html#ctranslate2.Translator.translate_batch
        params = {
        'beam_size' : self.beam_size,
        'patience' : self.patience,
        'num_hypotheses' : self.num_hypotheses,
        'length_penalty' : self.length_penalty,
        'coverage_penalty' : self.coverage_penalty,
        'repetition_penalty' : self.repetition_penalty,
        'no_repeat_ngram_size' : self.no_repeat_ngram_size,
        'disable_unk' : self.disable_unk,
        'suppress_sequences' : self.suppress_sequences,
        'end_token' : self.end_token,
        'return_end_token' : self.return_end_token,
        'prefix_bias_beta' : self.prefix_bias_beta,
        'max_input_length' : self.max_input_length,
        'max_decoding_length' : self.max_decoding_length,
        'min_decoding_length' : self.min_decoding_length,
        'use_vmap' : self.use_vmap,
        'return_scores' : self.return_scores,
        'return_attention' : self.return_attention,
        'return_alternatives' : self.return_alternatives,
        'min_alternative_expansion_prob' : self.min_alternative_expansion_prob,
        'sampling_topk' : self.sampling_topk,
        'sampling_topp' : self.sampling_topp,
        'sampling_temperature' : self.sampling_temperature,
        'replace_unknowns' : self.replace_unknowns,
        }
        if self.engineMajorVersion >= 4:
            if self.engineMinorVersion >= 4:
                # return_logits_vocab was added to ctranslate2.Translator().translate_batch() as a parameter starting at version 4.4.0, so only include it for that version or newer.
                params[ 'return_logits_vocab' ] = self.return_logits_vocab

        # ** means unpack the dictionary to key=value pairs suitable for use as function keyword parameters. One * means to unpack the dictionary's keynames and use them as positional parameters, not the value of the keys.
        translatedList = self.translator.translate_batch( encodedList, **params )
        return self.decode( translatedList, sourceLanguage, targetLanguage )


    def decode( self, translatedList, sourceLanguage=None, targetLanguage=None ):
        decodedList = [ ]
        for i in range( len( translatedList ) ):
            decodedList.append( self.targetLanguageTokenizer.decode( translatedList[ i ].hypotheses[ 0 ] ) )
        return decodedList


    # This returns the spm/ folder path based on C:/path/to/model.bin
    def _getSpmPath( self, modelFolderOnly ):
        guessedSourceSpmName = sentencePieceModelPrefix + defaultSourceLanguage + sentencePieceModelPostfix
        guessedTargetSpmName = sentencePieceModelPrefix + defaultTargetLanguage + sentencePieceModelPostfix
        foundSourceSpm = None
        foundTargetSpm = None

        if pathlib.Path( modelFolderOnly + '/' + guessedSourceSpmName ).is_file() == True:
            foundSourceSpm = modelFolderOnly + '/' + guessedSourceSpmName
        elif pathlib.Path( modelFolderOnly + '/../' + guessedSourceSpmName ).is_file() == True:
            foundSourceSpm = modelFolderOnly + '/../' + guessedSourceSpmName

        if pathlib.Path( modelFolderOnly + '/' + guessedTargetSpmName ).is_file() == True:
            foundTargetSpm = modelFolderOnly + '/' + guessedTargetSpmName
        elif pathlib.Path( modelFolderOnly + '/../' + guessedTargetSpmName ).is_file() == True:
            foundTargetSpm = modelFolderOnly + '/../' + guessedTargetSpmName

        for folder in self.sentencePieceModelFolderDictionary:
            if foundSourceSpm == None:
                if pathlib.Path( modelFolderOnly + '/' + folder + '/' + guessedSourceSpmName ).is_file() == True:
                    foundSourceSpm = modelFolderOnly + '/' + folder + '/' + guessedSourceSpmName
                elif pathlib.Path( modelFolderOnly + '/../' + folder + '/' + guessedSourceSpmName ).is_file() == True:
                    foundSourceSpm = modelFolderOnly + '/../' + folder + '/' + guessedSourceSpmName
            if foundTargetSpm == None:
                if pathlib.Path( modelFolderOnly + '/' + folder + '/' + guessedTargetSpmName ).is_file() == True:
                    foundTargetSpm = modelFolderOnly + '/' + folder + '/' + guessedTargetSpmName
                elif pathlib.Path( modelFolderOnly + '/../' + folder + '/' + guessedTargetSpmName ).is_file() == True:
                    foundTargetSpm = modelFolderOnly + '/../' + folder + '/' + guessedTargetSpmName

        if foundSourceSpm != None:
            foundSourceSpm = commonFunctions.fixPath( foundSourceSpm, basePath=modelFolderOnly )
        if foundTargetSpm != None:
            foundTargetSpm = commonFunctions.fixPath( foundTargetSpm, basePath=modelFolderOnly )

        print( ( 'Set sourceLanguageSentencePieceModel to \'' + str( foundSourceSpm ) + '\'' ).encode( consoleEncoding ) )
        print( ( 'Set targetLanguageSentencePieceModel to \'' + str( foundTargetSpm ) + '\'' ).encode( consoleEncoding ) )

        return foundSourceSpm, foundTargetSpm


        if (sourceLanguage == None) and (checkIfThisFileExists(sourceSentencePieceModel) != True):
            sys.exit ('Please specify a source language or a valid sourceSentencePieceModel.')
        if (targetLanguage == None) and (checkIfThisFileExists(targetSentencePieceModel) != True):
            sys.exit ('Please specify a target language or a valid targetSentencePieceModel.')

        #So the sentence piece source model is always required. For ctranslate2 both source and target models are both required. If not present, then try to use the defaults and/or the specified language to guess them.
        if checkIfThisFileExists(sourceSentencePieceModel) == True:
            # if a source language was not specified, try to guess source language based upon source sentencepiece model.
            if (sourceLanguage == None):
                #sourceSentencePieceModelPathObject = pathlib.Path(sourceSentencePieceModel).absolute()
                #sourceSentencePieceModelNameOnly = sourceSentencePieceModelPathObject.name
                sourceSentencePieceModelNameOnly = pathlib.Path(sourceSentencePieceModel).name

                # check to make sure both prefix and post fix are found in sourceSentencePieceModelNameOnly
                # error out if either of them are not found because they must both be present
                if (sourceSentencePieceModelNameOnly.find( defaultSentencePieceModelPrefix ) == -1) or ( sourceSentencePieceModelNameOnly.find( defaultSentencePieceModelPostfix ) == -1):
                    sys.exit('Unable to determine source language from sentencepiece model name. Please specify --sourceLanguage (-sl).' + usageHelp)

                # Remove prefix and postfix from the name. 
                tempString=sourceSentencePieceModelNameOnly.replace(defaultSentencePieceModelPrefix,'')
                tempString=tempString.replace(defaultSentencePieceModelPostfix,'')

                #If the result is not length = 2 or length=3, then error out,
                if ( len(tempString) <=1 ) or ( len(tempString) >= 4):
                    sys.exit('Unable to determine source language from sentencepiece model name. Incorrect length. Please specify --sourceLanguage (-sl).' + usageHelp)

                #otherwise set source language to those two or three characters.
                sourceLanguage=tempString

                print( ('Set sourceLanguage to \'' + sourceLanguage + '\' from: \'' + sourceSentencePieceModelNameOnly + '\'.').encode(consoleEncoding) )

        #if checkIfThisFileExists(sourceSentencePieceModel) != True:
        else: 
            tempFileName=defaultSentencePieceModelPrefix+sourceLanguage+defaultSentencePieceModelPostfix
            #tempPath=modelFolderOnly
            if checkIfThisFileExists(modelFolderOnly + '/' + tempFileName) == True:
                sourceSentencePieceModel=modelFolderOnly + '/' + tempFileName
            elif checkIfThisFileExists(modelFolderOnly + '/../' + tempFileName) == True:
                sourceSentencePieceModel=modelFolderOnly + '/../' + tempFileName
            elif checkIfThisFileExists(modelFolderOnly+ '/' + defaultSentencePieceModelFolder0 + '/' + tempFileName) == True:
                sourceSentencePieceModel=modelFolderOnly+ '/' + defaultSentencePieceModelFolder0 + '/' + tempFileName
            elif checkIfThisFileExists(modelFolderOnly+ '/' + defaultSentencePieceModelFolder1 + '/' + tempFileName) == True:
                sourceSentencePieceModel=modelFolderOnly+ '/' + defaultSentencePieceModelFolder1 + '/' + tempFileName
            elif checkIfThisFileExists(modelFolderOnly+ '/' + defaultSentencePieceModelFolder2 + '/' + tempFileName) == True:
                sourceSentencePieceModel=modelFolderOnly+ '/' + defaultSentencePieceModelFolder2 + '/' + tempFileName
            elif checkIfThisFileExists(modelFolderOnly+ '/../' + defaultSentencePieceModelFolder0 + '/' + tempFileName) == True:
                sourceSentencePieceModel=modelFolderOnly+ '/../' + defaultSentencePieceModelFolder0 + '/' + tempFileName
            elif checkIfThisFileExists(modelFolderOnly+ '/../' + defaultSentencePieceModelFolder1 + '/' + tempFileName) == True:
                sourceSentencePieceModel=modelFolderOnly+ '/../' + defaultSentencePieceModelFolder1 + '/' + tempFileName
            elif checkIfThisFileExists(modelFolderOnly+ '/../' + defaultSentencePieceModelFolder2 + '/' + tempFileName) == True:
                sourceSentencePieceModel=modelFolderOnly+ '/../' + defaultSentencePieceModelFolder2 + '/' + tempFileName
            verifyThisFileExists( sourceSentencePieceModel, 'sourceSentencePieceModel' )

            if __name__ == '__main__':
                print( ('Set sourceSentencePieceModel to \'' + str(sourceSentencePieceModel) + '\' from: \'' + sourceLanguage + '\'.').encode(consoleEncoding) )

        if checkIfThisFileExists(targetSentencePieceModel) == True:
            #If a target language was not specified, try to guess target language based upon target sentencepiece model.
            if (targetLanguage == None):
                targetSentencePieceModelNameOnly = pathlib.Path(targetSentencePieceModel).name

                # check to make sure both prefix and post fix are found in targetSentencePieceModelNameOnly
                # error out if either of them are not found because they must both be present
                if (targetSentencePieceModelNameOnly.find( defaultSentencePieceModelPrefix ) == -1) or ( targetSentencePieceModelNameOnly.find( defaultSentencePieceModelPostfix ) == -1):
                    sys.exit('Unable to determine target language from sentencepiece model name. Please specify --targetLanguage (-sl).' + usageHelp)

                # Remove prefix and postfix from the name. 
                tempString=targetSentencePieceModelNameOnly.replace(defaultSentencePieceModelPrefix,'')
                tempString=tempString.replace(defaultSentencePieceModelPostfix,'')

                #If the result is not length = 2 or length=3, then error out,
                if ( len(tempString) <=1 ) or ( len(tempString) >= 4):
                    sys.exit('Unable to determine target language from sentencepiece model name. Incorrect length. Please specify --targetLanguage (-sl).' + usageHelp)

                #otherwise set target language to those two or three characters.
                targetLanguage=tempString

                if __name__ == '__main__':
                    print( ('Set targetLanguage to \'' + targetLanguage + '\' from: \'' + targetSentencePieceModelNameOnly + '\'.').encode(consoleEncoding) )

        #if checkIfThisFileExists(targetSentencePieceModel) != True
        else:
            tempFileName=defaultSentencePieceModelPrefix+targetLanguage+defaultSentencePieceModelPostfix
            #tempPath2=modelFolderOnly + '/' + tempFileName
            if checkIfThisFileExists(modelFolderOnly + '/' + tempFileName) == True:
                targetSentencePieceModel=modelFolderOnly + '/' + tempFileName
            elif checkIfThisFileExists(modelFolderOnly + '/../' + tempFileName) == True:
                targetSentencePieceModel=modelFolderOnly + '/../' + tempFileName
            elif checkIfThisFileExists(modelFolderOnly + '/' + defaultSentencePieceModelFolder0 + '/' + tempFileName) == True:
                targetSentencePieceModel=modelFolderOnly + '/' + defaultSentencePieceModelFolder0 + '/' + tempFileName
            elif checkIfThisFileExists(modelFolderOnly + '/' + defaultSentencePieceModelFolder1 + '/' + tempFileName) == True:
                targetSentencePieceModel=modelFolderOnly + '/' + defaultSentencePieceModelFolder1 + '/' + tempFileName
            elif checkIfThisFileExists(modelFolderOnly + '/' + defaultSentencePieceModelFolder2 + '/' + tempFileName) == True:
                targetSentencePieceModel=modelFolderOnly + '/' + defaultSentencePieceModelFolder2 + '/' + tempFileName
            elif checkIfThisFileExists(modelFolderOnly + '/../' + defaultSentencePieceModelFolder0 + '/' + tempFileName) == True:
                targetSentencePieceModel=modelFolderOnly + '/../' + defaultSentencePieceModelFolder0 + '/' + tempFileName
            elif checkIfThisFileExists(modelFolderOnly + '/../' + defaultSentencePieceModelFolder1 + '/' + tempFileName) == True:
                targetSentencePieceModel=modelFolderOnly + '/../' + defaultSentencePieceModelFolder1 + '/' + tempFileName
            elif checkIfThisFileExists(modelFolderOnly + '/../' + defaultSentencePieceModelFolder2 + '/' + tempFileName) == True:
                targetSentencePieceModel=modelFolderOnly + '/../' + defaultSentencePieceModelFolder2 + '/' + tempFileName
            #The target is optional for fairseq, but required for ctranslate2.
            if mode == 'ctranslate2':
                verifyThisFileExists(targetSentencePieceModel,'targetSentencePieceModel')


