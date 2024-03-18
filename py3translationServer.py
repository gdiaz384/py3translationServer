#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Description:
py3translationServer.py exposes fairseq and CTranslate2 models over HTTP using the Tornado web server.

- Tornado is a Python web framework and asynchronous networking library with an emphasis on non-blocking network I/O.
- fairseq is library for machine learning and data modeling.
- CTranslate2 is a C++ and Python library for efficient inference with transformer models, including those used by fairseq.
- More information:
    - https://www.tornadoweb.org
    - https://github.com/facebookresearch/fairseq
    - https://opennmt.net/CTranslate2

Install with:
- pip install tornado ctranslate2
- fairseq must be built from source since the 0.2.0 version available on PyPi is too old.

py3translationServer.py:
- Supports both CPU and GPU inferencing. 'GPU' is aliased to CUDA, but DirectML is also supported on Windows.
- Supports large batch requests.
- Supports both single process and multiprocess modes.
    - In single process mode, the model is preloaded for low latency inferencing.
    - In multiprocess mode, the model has significantly longer initial startup time (5 seconds+) but returns all memory allocated once the transaction completes. This is ideal for batch translations and long term operation.

Copyright: github/gdiaz384
License: AGPLv3, https://www.gnu.org/licenses/agpl-3.0.html
"""
#import multiprocessing
#if ( __name__ == '__main__' ):
#    multiprocessing.freeze_support()  # Does not work.

__version__ = '0.4 beta - 2024Feb24' #This should probably be __version__ by convention. 'version' by itself is wrong since there is a conflicting '--version' CLI option that must not be changed because then it would change the UI of the CLI. #Update. Changed it.


# Set global defaults:
defaultFileEncoding='utf-8'
defaultConsoleEncoding='utf-8'
#https://docs.python.org/3.8/library/codecs.html#error-handlers
defaultInputFileErrorHandling='strict'


# Set main program defaults:
# Valid values: cpu, gpu, cuda, directml. gpu is aliased to cuda.
# ROCm support is not currently implemented. Entering 'rocm' will use fairseq in CPU mode and error out CTranslate2.
defaultDevice='cpu'

# Use two letter language codes: www.loc.gov/standards/iso639-2/php/code_list.php
# Currently unused. Source and target languages must be specified at runtime.
defaultSourceLanguage='ja'
defaultTargetLanguage='en'

# This is relative to inputModelOrFolder which must be specified at the command prompt.
# Example sentence pieces: https://huggingface.co/JustFrederik
defaultSentencePieceModelFolder0='spm'
defaultSentencePieceModelFolder1='spmModel'
defaultSentencePieceModelFolder2='spmModels'
defaultSentencePieceModelPrefix='spm.'
defaultSentencePieceModelPostfix='.nopretok.model'
# If no sourceSentencePieceModel is specified, then use the defaultSentencePieceModelFolder0 together with defaultSourceLanguage to compute a value for sourceSentencePieceModel and check if it exists as a file. If it exists, use it. Example:
#'spm.ja.nopretok.model'
# If no targetSentencePieceModel is specified, then use the defaultSentencePieceModelFolder0 together with defaultTargetLanguage to compute a value for targetSentencePieceModel and check if it exists as a file. If it exists, use it. Example:
#'spm.en.nopretok.model'

defaultCTranslate2ModelName='model.bin'

# Host address and port. 0.0.0.0 means 'bind to all local addresses'.
#defaultAddress='0.0.0.0'
defaultAddress='localhost'  # localhost has an alias of 127.0.0.1
defaultPort=14366

# The amount of time, in seconds, that must pass before the next request will trigger writing the cache to disk. Set to low value, like 1 to nearly always write out file.
#  In some situations, writing the file may take several seconds. A safe minimum amount should be ~10 assuming a healthy disk and low to moderate active I/O.
defaultSaveCacheInterval=60

# The minumum time to wait in between allowing cache to be cleared meaning that cache cannot be cleared within this window of writing it out.
# Not implemented yet.
defaultMinimumClearCacheInterval=60

# Valid values are True or False. Default=True. Set to False to overwrite cache.csv in-place without creating a copy. Not implemented yet.
defaultCreateBackupOfCacheFile=True

# This is relative to path of main script or the local environment. TODO: The path handling logic should be updated to not break if an absolute path is entered here.
defaultCacheLocation='resources/cache'

# defaultCacheLocation is normally used to store cache. Setting the following to True changes the storage location of the cache to:
    # Windows: os.getenv('LOCALAPPDATA') / py3translationServer/cache
    # Linux: ~/.config/py3translationServer/cache
# Not implemented yet.
defaultStoreCacheInLocalEnvironment=False

# Valid values are spawn, fork, and forkserver. Changing this will lead to untested behavior.
# https://docs.python.org/3.12/library/multiprocessing.html#contexts-and-start-methods
defaultProcessesSpawnTechnique='spawn'
# fairseq does not play well with multithreading or multiprocessing, so create a toggle to help troubleshooting.
defaultfairseqMultithreadingEnabled=True


# These are internal variable names for fairseq and CTranslate2, so they use a slightly different variable naming scheme.
# Fairseq documentation and source code:
# https://fairseq.readthedocs.io/en/latest/models.html#fairseq.models.transformer.TransformerModel
# https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_base.py
# https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_legacy.py
# https://fairseq.readthedocs.io/en/latest/_modules/fairseq/models/fairseq_model.html#BaseFairseqModel.from_pretrained
# https://fairseq.readthedocs.io/en/latest/_modules/fairseq/tasks/translation.html?highlight=source_lang
# https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-interactive

# CTranslate2 documentation and source code:
# https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html

# Valid bpe values are: byte_bpe, bytes, characters, fastbpe, gpt2, bert, hf_byte_bpe, sentencepiece, subword_nmt
# Depends upon model/model format used.
# Note that OpenNMT refers to this as as the tokenizer type but fairseq uses a different tokenizer concept for their UI: moses, nltk, space. This default_bpe uses the value options defined by fairseq.
default_bpe='sentencepiece'

# CTranslate2 documentation:
# https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html
# https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html#ctranslate2.Translator.translate_batch

# Maximum number of parallel translations. Higher values affect video memory usage. Seems to have no or little effect on CPU loads and processing time.
default_inter_threads=16

# Number of OpenMP CPU threads per translator (0 to use a default value). if the psutil library is available, then this will be updated dynamically.
default_intra_threads=0

# https://fairseq.readthedocs.io/en/latest/_modules/fairseq/tasks/fairseq_task.html?highlight=beam_size
#beam_size is the number of tokens generated by the model. The best one will be chosen as the return value. Directly affects quality. This is the main speed vs quality setting.
# CTranslate2 default=2. Changed to 5 as per default setting in fairseq source code. Set beam size (1 for greedy search). Best performance is 1.
default_beam_size=5

# Number of results to return.
default_num_hypotheses=1
default_no_repeat_ngram_size=3
# Setting this to True corrupts the output, so leave as False until correct vmap can be built. Update: Added this to CLI instead.
#default_use_vmap=False


#Might be an interesting read: https://docs.python.org/3/library/configparser.html
import argparse                # Used to add command line options.
import sys                        # End program on fail condition. Technically, this always exits as an error for anything but sys.exit(0) even if just trying to close normally, but w/e.
import os                          # Test if file/folder exists.
#import io, iostream, gen  # Optional. Read from and write to objects in memory as if they were files. Used for sending cache.csv directly from memory and perhaps will be used later for cache.csv.zip. Not fully implemented yet. Import conditionally later if needed.
#import socket                   # Optional. Used to get IP's and print them to clarify to the user where Tornado is listening. Import as needed.
import pathlib                   # Part of standard library since 3.4. Imported for Path class which does sane path handling.
import json                       # Accept JSON as input. Return JSON after processing.
import time                      # Optional library. Used to calculate performance metrics. Import conditionally later. #Update: perfMetrics, cache write out time, and clear cache time require this, so just always include it instead. Part of standard library anyway.
#import csv                        # Used to read/write cache files. Import conditionally later based upon if cache is enabled or not.
#import date or datetime   # Humm. Could be used to append the current date to the cache backup file as cache.hash.csv.backup.Today.csv
import signal                   #Sometimes required library. This is needed to send signal.SIGTERM to terminate processes when fairseq + CPU hangs. import conditionally as needed. Also used for UI.
#import inspect               #Used to print out the name of the current function during execution which is useful when debugging. Import conditionally later.
import hashlib                 # Used to identify correct cache.csv on disk and also as a psudo-rng function for temporary writes.

#import fairseq                 # Core engine. Must be installed with 'pip install fairseq' or built from source. Import conditionally later.
#import ctranslate2           # Core engine. Must be installed with 'pip install ctranslate2'. Import conditionally later.
#import sentencepiece      # Core dependency. Must be installed with 'pip install sentencepiece' Used for both fairseq and ctranslate2. However, fairseq will import it internally, like with PyTorch, so do not worry about it explicitly unless ctranslate2 is specified.

import asyncio                # Used for asynconous I/O. Part of standard library since 3.4. Is also a tornado dependency.
import multiprocessing     # Part of standard library. Used for Process, Queue objects. Used in core logic and also in cache logic. #Should probably import conditionally. #Update, this is still needed, even with concurrent.futures, to deterministically set the spawn type for the child processes, spawn, fork, forserver, but is still technically optional if cache is not enabled and if preloadModel==True. Annoying to import conditionally.
import concurrent.futures # Used to create a process that can work with asynconous I/O. Basically asyncio + multiprocessing.
import tornado                 # Web server. tornado.escape.json_decode creates Python dictionary from input json. Must be installed with 'pip install tornado'.
import tornado.web          # This duplicate explicit import improves compatibility with Python versions < 3.8 and pyinstaller.
try:
    import psutil                 # This library is required for fairseq + CPU + multiprocessing, but technically optional otherwise. This library is also used to optimize CTranslate2 to use the number of physical cores if running on CPU. #Update: It should be possible to remove this requirement by altering the way the new process returns its data to always return the process ID. However, the signal library would still be required and sending signal.SIGTERM to the process might be more complicated, os specific, or unsafe. Update: This is also used to identify and child processes when launching the UI in order to close them during shutdown, so back in required territory.
    psutilAvailable=True
except ImportError:
    psutilAvailable=False


# Set some more defaults that need to be after the import statments.
currentScriptNameWithoutPath=str( os.path.basename(str(__file__)) )
usageHelp=' Usage: ' + currentScriptNameWithoutPath + ' -h'
defaultSysCacheLocationWin=os.getenv('LOCALAPPDATA')
defaultSysCacheLocationWin='~/.local/'+currentScriptNameWithoutPath


# Update ctrl + c handler on Windows. Linux should work mostly as expected without modification.
# From Shital Shah at https://stackoverflow.com/questions/1364173/stopping-python-using-ctrlc
# Had to change b=None to no default value, but ctrl+c seems to work more reliably now. Maybe. Still does not work sometimes.
# The only workaround might be to always launch the .py from its own .cmd and then tell cmd to close.
# The b in handler also does not always work but setting a default is also error prone.
# install with: pip install pywin32
#def handler(a,b):
#    sys.exit(0)
if sys.platform == 'win32':
    try:
        # Load different handler function for different Python versions to sometimes improve compatibility for older versions.
        # This maybe sometimes breaks compatibility for newer Python versions, maybe.
        if int(sys.version_info[1]) >= 8:
            def handler(a,b):
                sys.exit(0)
        else:
            def handler(a,b=None):
                sys.exit(0)
        import win32api
        win32api.SetConsoleCtrlHandler(handler, True)
    except ImportError:
        pass


# Add command line options.
commandLineParser=argparse.ArgumentParser(description='Description: '+ currentScriptNameWithoutPath + ' exposes fairseq and CTranslate2 models over HTTP using the Tornado web server. ' + usageHelp)

commandLineParser.add_argument('mode', help='Must be fairseq or ctranslate2.', default=None, type=str)
commandLineParser.add_argument('modelPath', help='For fairseq, the model.pretrain, including path. For CTranslate2, the folder containing model.bin.', default=None, type=str)

commandLineParser.add_argument('-dev', '--device', help='Process using cpu, gpu, cuda, or directml. gpu is aliased to cuda. rocm is not supported yet. Default='+defaultDevice, default=defaultDevice, type=str)

commandLineParser.add_argument('-sl', '--sourceLanguage', help='Two letter source language code. See: www.loc.gov/standards/iso639-2/php/code_list.php Default=None', default=None, type=str)
commandLineParser.add_argument('-tl', '--targetLanguage', help='Two letter target language code. See: www.loc.gov/standards/iso639-2/php/code_list.php Default=None', default=None, type=str)
commandLineParser.add_argument('-sspm', '--sourceSentencePieceModel', help='The source sentencepiece model name and path. Default is based on source language.', default=None, type=str)
commandLineParser.add_argument('-tspm', '--targetSentencePieceModel', help='The target sentencepiece model and path. Default is based on target language.', default=None, type=str)

commandLineParser.add_argument('-pm', '--preloadModel', help='Make the system run out of memory. Default=Disabled.', action='store_true')
commandLineParser.add_argument('-t', '--cpuThreads', help='Specify the number of CPU threads. Only affects CTranslate2. If the psutil library is available, the default is the number of physical cores. Otherwise without psutil, CTranslate2 will use its internal values. Using psutil requires installing it via: pip install psutil', default=None, type=int)
commandLineParser.add_argument('-vm', '--useVMap', help='For CTranslate2, enabe the use of a vocabulary map. Must be named vmap.txt. Default=False.', action='store_true')
commandLineParser.add_argument('-dpm', '--disablePerfMetrics', help='Disable tracking and reporting of performance metrics. Default=Enabled.', action='store_false')

commandLineParser.add_argument('-c', '--cache', help='Toggle cache setting from default. Enabling cache saves the results of the model for future requests. Default=cache is enabled.', action='store_false')
commandLineParser.add_argument('-ui', '--uiPath', help='Specify the path to the streamlit UI. Using streamlit requires installing it via: pip install streamlit', default=None, type=str)

commandLineParser.add_argument('-a', '--address', help='Specify the address to listen on. To bind to all addresses, use 0.0.0.0  Default is to bind to: '+ str(defaultAddress), default=defaultAddress, type=str)
commandLineParser.add_argument('-p', '--port', help='Specify the port the local server will use. Default=' + str(defaultPort), default=defaultPort, type=int)

commandLineParser.add_argument('-cfe', '--cacheFileEncoding', help='Specify the encoding used for cache.csv. Default='+defaultFileEncoding,default=defaultFileEncoding, type=str)
commandLineParser.add_argument('-ce', '--consoleEncoding', help='Specify the encoding used for certain types of stdout. Default='+defaultConsoleEncoding,default=defaultConsoleEncoding, type=str)
commandLineParser.add_argument('-ifeh', '--inputFileErrorHandling', help='If the input from files cannot be read perfectly using the specified encoding, what should happen? See: https://docs.python.org/3.8/library/codecs.html#error-handlers Default is to crash the program.', default=defaultInputFileErrorHandling, type=str)
commandLineParser.add_argument('-v', '--version', help='Print version information and exit.', action='store_true')
commandLineParser.add_argument('-vb', '--verbose', help='Print more information.', action='store_true')
commandLineParser.add_argument('-d', '--debug', help='Print too much information.', action='store_true')


# Parse command line settings.
commandLineArguments=commandLineParser.parse_args()

mode=commandLineArguments.mode
inputModelFileOrFolder=commandLineArguments.modelPath

device=commandLineArguments.device
sourceLanguage=commandLineArguments.sourceLanguage
targetLanguage=commandLineArguments.targetLanguage
sourceSentencePieceModel=commandLineArguments.sourceSentencePieceModel
targetSentencePieceModel=commandLineArguments.targetSentencePieceModel

preloadModel=commandLineArguments.preloadModel
intra_threads=commandLineArguments.cpuThreads
use_vmap=commandLineArguments.useVMap
perfMetrics=commandLineArguments.disablePerfMetrics

cacheEnabled=commandLineArguments.cache
uiPath=commandLineArguments.uiPath

address=commandLineArguments.address
port=commandLineArguments.port

cacheFileEncoding=commandLineArguments.cacheFileEncoding
consoleEncoding=commandLineArguments.consoleEncoding
inputErrorHandling=commandLineArguments.inputFileErrorHandling
version=commandLineArguments.version
verbose=commandLineArguments.verbose
debug=commandLineArguments.debug


# Validate input.
if (perfMetrics == True) or (verbose==True) or (debug == True):
    #import time                     # Optional library. Used to calculate performance metrics. #Update, processing time should be optionally reported even if verbose==True, so load it if either of those conditions are true. Debug being true implies that verbose is as well. # Update2. Will need to always import time at some point for cache functionality for delaying writing out cache file for at least 30s, ideally 60s.
    startedLoadingTime = time.perf_counter()


if version == True:
    sys.exit( (currentScriptNameWithoutPath + ' ' + __version__).encode(consoleEncoding) )


if debug == True:
    verbose = True
    import inspect   #Used to print out the name of the current function during execution which is useful when debugging.


# Define helper functions to help validate input.
def verifyThisFileExists(myFile,nameOfFileToOutputInCaseOfError=None):
    if myFile == None:
        sys.exit( ('Error: Please specify a valid file for: ' + str(nameOfFileToOutputInCaseOfError) + usageHelp).encode(consoleEncoding))
    if os.path.isfile(myFile) != True:
        sys.exit( (' Error: Unable to find file \'' + str(nameOfFileToOutputInCaseOfError) + '\' ' + usageHelp).encode(consoleEncoding) )
 
def verifyThisFolderExists(myFolder, nameOfFileToOutputInCaseOfError=None):
    if myFolder == None:
        sys.exit( ('Error: Please specify a valid folder for: ' + str(nameOfFileToOutputInCaseOfError) + usageHelp).encode(consoleEncoding))
    if os.path.isdir(myFolder) != True:
        sys.exit( (' Error: Unable to find folder \'' + str(nameOfFileToOutputInCaseOfError) + '\' ' + usageHelp).encode(consoleEncoding) )

def checkIfThisFileExists(myFile):
    if (myFile == None) or (os.path.isfile(myFile) != True):
        return False
    return True

def checkIfThisFolderExists(myFolder):
    if (myFolder == None) or (os.path.isdir(myFolder) != True):
        return False
    return True


#Update path of current script.
currentScriptPathObject = pathlib.Path( __file__ ).absolute()
currentScriptPathOnly = str(currentScriptPathObject.parent) #Does not include last / and this will return one subfolder up if it is called on a folder.
#currentScriptNameWithoutPath=    #This was defined earlier already


inputModelFileNameAndPath=None
inputModelPathOnly=None
inputModelNameWithoutPath=None
# mode and inputModel will always be used at the CLI as required inputs, so just need to validate they are correct.
# mode must be fairseq or CTranslate2
if mode.lower() == 'fairseq':
    try:
        import fairseq
    except ImportError:
        sys.exit( 'Error: fairseq was selected for mode but cannot be imported. Please install it with: pip install fairseq' )

    mode = 'fairseq'

    # inputModelFileOrFolder must be a file and it must exist
    verifyThisFileExists( inputModelFileOrFolder , 'inputModelFileOrFolder' )
    #If there is a folder specified, could also try to auto detect a pretrained.pt model for increased flexibility.

    # Create subtypes here using Path library, like path only, extension only. Not sure how they will be used/useful, but can just comment out later.
    inputModelFileNameAndPath=inputModelFileOrFolder
    inputModelPathObject= pathlib.Path( inputModelFileNameAndPath ).absolute()
    inputModelPathOnly = str(inputModelPathObject.parent) # Does not include last /, and this will return one subfolder up if it is called on a folder.
    inputModelNameWithoutPath = inputModelPathObject.name
elif mode.lower() == 'ctranslate2':
    try:
        import ctranslate2
    except ImportError:
        sys.exit( 'Error: ctranslate2 was selected for mode but cannot be imported. Please install it with: pip install ctranslate2' )
    try:
        import sentencepiece
    except ImportError:
        sys.exit( 'Error: sentencepiece cannot be imported. Please install sentencepiece with: pip install sentencepiece' )

    mode = 'ctranslate2'
    inputModelFileOrFolderObject=pathlib.Path( inputModelFileOrFolder ).absolute()

    # If the specified path is a file, then get the folder from the str(pathlib.Path(myPath).parent)
    # and then continue to run as normal. ctranslate2 will refuse to load the model if not valid, so do not worry about it.
    if checkIfThisFileExists(inputModelFileOrFolder) == True:
        inputModelFileNameAndPath=str(inputModelFileOrFolderObject)
        inputModelPathOnly=str(inputModelFileOrFolderObject.parent)
        inputModelNameWithoutPath=inputModelFileOrFolderObject.name
    else:
        #inputModelFileOrFolder must be a folder and it must exist
        # The model must also exist inside of it, but maybe let the ctranslate2 library worry about that? It might have its own code for detecting different ctranslate2 formats or w/e.
        verifyThisFolderExists(inputModelFileOrFolder,'inputModelFileOrFolder')

        # Create subtypes here using Path library.
        inputModelFileNameAndPath=str(inputModelFileOrFolderObject) + '/' + defaultCTranslate2ModelName
        inputModelPathOnly=str(inputModelFileOrFolderObject)
        # if no model name was specified, then fudge the model name based upon last folder in the path. #Might want to just set this to the defaultCTranslate2ModelName instead.
        inputModelNameWithoutPath=inputModelFileOrFolderObject.parts[len(inputModelFileOrFolderObject.parts)-1]
else:
    sys.exit( ('Error: mode must be ctranslate2 or fairseq. Mode=' + str(mode)).encode(consoleEncoding) )


# Now that inputModelNameWithoutPath is known, update some more variables for later use.
scriptNameWithVersion = currentScriptNameWithoutPath + '/' +__version__
scriptNameWithVersionDictionary = { 'content' : scriptNameWithVersion }
modeAndModelName = mode + '/' + inputModelNameWithoutPath
modeAndModelNameDictionary = { 'content' : modeAndModelName }


# verify device
if device.lower() == 'cpu':
    device='cpu'
elif device.lower() == 'gpu':
    # Create alias.
    device='cuda'
elif device.lower() == 'cuda':
    device='cuda'
elif device.lower() == 'rocm':
    device='rocm'
elif device.lower() == 'directml':
    device='directml'
    if mode != 'fairseq':
        sys.exit( ('Error: Device \'directml\' is only valid for fairseq. Mode=\''+ mode + '\' Current device=\'' + device +'\'').encode(consoleEncoding) )
    try:
        # https://github.com/microsoft/DirectML/tree/master/PyTorch/1.13
        import torch
        import torch_directml
        dml = torch_directml.device()
    except ImportError:
        sys.exit( 'Problem avoided: directml was specified but did not import sucessfully. Consider using anything else, like ctranslate2. Installing directml will trash any existing PyTorch installation. Do not use. Alternatively: pip install torch-directml')
else:
    sys.exit( ('Error: Unrecognized device=\'' + device + '\' Must be cpu, gpu, cuda, rocm, or directml.').encode(consoleEncoding) )


# Update cache path and related settings.
# lazyHash is for use in a different process so that the main process does not get overfilled with memory that it will never again use when the entire model contents are read into memory.
# Proper way is probably to create a thread and read the file in chunks, but since the model is expected to be in memory later on anyway, reading it all in at once does not bloat the memory requirements of this program beyond what they already are. However, not reading it in either in another process, or in chunks would bloat the size.
def lazyHash(fileNameAndPath,myQueue):
    # SHA1
    with open(inputModelFileNameAndPath,'rb') as myFile:
        myFileContents=myFile.read()
        #modelHash=str(hashlib.sha1(myFileContents).hexdigest())[:10]
    myQueue.put( str( hashlib.sha1(myFileContents).hexdigest() ) )

    # CRC32
    # So, this returns a different crc32 than 7-Zip regardless of binascii/zlip or the 'bitwise and' fix.
    # Apparently, there are different sub standards for CRC32.
    # https://reveng.sourceforge.io/crc-catalogue/all.htm
    # Since SHA1 is too long, CRC32 is just borked, and there are no CRC64 libs in the Python standard library, just use a trunkated SHA1 hash as a compromise. Quirky, but whatever.
    #import zlib
    #import binascii
    #with open(inputModelFileNameAndPath,'rb') as myFile:
    #    myFileContents=myFile.read()
    #    #modelHash=binascii.crc32(myFileContents)
    #myQueue.put(str( (zlib.crc32(myFileContents)) & 0xffffffff) )


# This turns translationCacheDictionary into a csv file at cacheFilePathAndName.
# That .csv can grow quite large, so support optional compression perhaps?
# https://docs.python.org/3/library/zipfile.html
# This UI is a bit odd. It should accept a dictionary and a fileNameAndPath. This should probably be turned into a Class that wraps a dictionary and handles the I/O.
def writeOutCache():
    # Spaghetti.
    global translationCacheDictionary
    global modelHashFull

    # Redundant, but it is better to be paranoid.
    pathlib.Path( cacheFilePathOnly ).mkdir( parents = True, exist_ok = True )

    #cacheFilePathOnly
    #cacheFilePathAndName is the final location
    #cacheFileNameOnly

    #hashlib.sha1(myFileContents).hexdigest()
    randomNumber=hashlib.sha1(cacheFilePathAndName.encode(consoleEncoding))
    randomNumber.update(str(time.perf_counter()).encode(consoleEncoding))
    randomNumber=str(randomNumber.hexdigest())[:8]
    temporaryFileNameAndPath=cacheFilePathOnly + '/' + 'cache.temp.' + randomNumber + '.csv'

    if debug == True:
        print( 'temporaryFileNameAndPath=' + temporaryFileNameAndPath )

    #write to temporary file first.
    with open(temporaryFileNameAndPath, 'w', newline='', encoding=cacheFileEncoding) as myOutputFileHandle:
        myCsvHandle = csv.writer(myOutputFileHandle)
        myCsvHandle.writerow(['rawText',inputModelNameWithoutPath + '.' +modelHashFull])
        for i, k in translationCacheDictionary.items():
            myCsvHandle.writerow( [str(i),str(k)] )

    if checkIfThisFileExists(temporaryFileNameAndPath) == True:
        #Replace any existing cache with the temporary one.
        pathlib.Path(temporaryFileNameAndPath).replace(cacheFilePathAndName)
        print( ('Wrote cache to disk at: ' + cacheFilePathAndName).encode(consoleEncoding) )
    else:
        print( ('Warning: Error writing temporary cache file at:' + temporaryFileNameAndPath).encode(consoleEncoding) )


#This turns translationCacheDictionary into a csv file at cacheFilePathAndName.
def clearCache():
    global translationCacheDictionary
    translationCacheDictionary={}
    print( 'Cleared cache.' )

if ( __name__ == '__main__' ) and ( cacheEnabled == True ):
    # import libraries specific to handling cache.
    import csv    #i/o cache to disk
    #import hashlib  # Used to identify correct cache.csv on disk and also as a psudo-rng function for temporary writes.

    # Initialize translationCacheDictionary
    translationCacheDictionary={}
    # Initalize timeCacheWasLastWritten
    timeCacheWasLastWritten=time.perf_counter()
    timeCacheWasLastCleared=time.perf_counter()

    verifyThisFileExists(inputModelFileNameAndPath,'modelNameAndPath')
    if debug == True:
        print('cacheEnabled='+str(cacheEnabled))
    print( 'Attempting to read cache for model: ' + str(inputModelFileNameAndPath) )

    # Dump the work of reading the file onto another process so main process does not have to deal with it.
    # This is low level code according to the concurrent.futures python docs since that documentation refers to itself as a high-level wrapper for multiprocessing.
    # https://docs.python.org/3/library/concurrent.futures.html
    # https://docs.python.org/3/library/multiprocessing.html
    modelHash=None
    myQueue = multiprocessing.Queue()
    lazyHashFunction = multiprocessing.Process(target=lazyHash, args=(inputModelFileNameAndPath,myQueue,) )
    lazyHashFunction.start()
    modelHashFull = myQueue.get()
    lazyHashFunction.join()
    if modelHashFull == None:
        sys.exit( ('Error: Could not generate hash from model file.' + str(inputModelFileNameAndPath)).encode(consoleEncoding) )
    modelHash=modelHashFull[:10] # Truncate hash to make the file name more friendly to file system length limitations.

    cacheFilePathOnly=currentScriptPathOnly+'/'+defaultCacheLocation
    cacheFileNameOnly='cache.'+ modelHash + '.csv' #Hardcoded. Maybe add prefix and postfix variables?
    cacheFilePathAndName=cacheFilePathOnly + '/' + cacheFileNameOnly

    if debug == True:#Maybe change this to debug for final settings.
        print( 'modelHash=' + str(modelHash) )
        print( 'cacheFilePathOnly=' + cacheFilePathOnly )
        print( 'cacheFileNameOnly=' + cacheFileNameOnly )
    if verbose == True:
        print( 'cacheFilePathAndName=' + cacheFilePathAndName )

    if checkIfThisFileExists(cacheFilePathAndName) ==  True:
        # Then cache exists. Path to it also already exists.
        # Read entries to translationCacheDictionary.
        # If valid then read as normal, but if any error occurs, then print out that there was an error when reading the cache file and just use a new one.
        try:
            with open(cacheFilePathAndName, newline='', encoding=cacheFileEncoding, errors=inputErrorHandling) as myFileHandle:
                csvReader = csv.reader(myFileHandle, strict=True)
                currentLine=0
                for line in csvReader:
                    #skip first line
                    if currentLine == 0:
                        currentLine+=1
                    elif currentLine != 0:
                        #if ignoreWhitespace == True:
                        for i in range(len(line)):
                            line[i]=line[i].strip()
                        if line[1] == '':
                            line[1] = None
                        translationCacheDictionary[line[0]]=line[1]
        except:
            print( ('Warning: Reinitalizing cache due to error reading input cache.csv: '+cacheFilePathAndName).encode(consoleEncoding) )
            translationCacheDictionary={}

        if debug == True:
            print( ('translationCacheDictionary=' + str(translationCacheDictionary)).encode(consoleEncoding) )

        print( 'Number of entries loaded into cache: ' + str(len(translationCacheDictionary)) )

        # Rename cache file to backup file regardless of I/O errors. File has already been verified to exist. Rename to backup.
        #print('pie')
        cacheBackupFileName=cacheFilePathAndName + '.backup'
        pathlib.Path(cacheFilePathAndName).replace(cacheBackupFileName) #It might make sense to append the date the backup was made, but could also just leave well enough alone.
        print ( ('Moved old cache.csv to: ' + cacheBackupFileName).encode(consoleEncoding) )

        # So, if the 'old' cache is moved and the still-in-memory-cache is never written out, then the cache will be deleted if the user does not translate at least 1 entry to trigger a cache write. To avoid that weird bug, flush cache here. This is a bit wasteful over just binary copying the file or not moving it until needed, but it also tests to make sure I/O actually works during initalization, so leave it.
        if len(translationCacheDictionary) > 0:
            writeOutCache()

    else:
        # Then cache does not exist. Create path. File will be created later when writing out entries.
        if verbose == True:
            print( (' Cache file not found. Creating a new one at: '+str(cacheFilePathAndName)).encode(consoleEncoding) )
        pathlib.Path( cacheFilePathOnly ).mkdir( parents = True, exist_ok = True )


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
    #tempPath=inputModelPathOnly
    if checkIfThisFileExists(inputModelPathOnly + '/' + tempFileName) == True:
        sourceSentencePieceModel=inputModelPathOnly + '/' + tempFileName
    elif checkIfThisFileExists(inputModelPathOnly + '/../' + tempFileName) == True:
        sourceSentencePieceModel=inputModelPathOnly + '/../' + tempFileName
    elif checkIfThisFileExists(inputModelPathOnly+ '/' + defaultSentencePieceModelFolder0 + '/' + tempFileName) == True:
        sourceSentencePieceModel=inputModelPathOnly+ '/' + defaultSentencePieceModelFolder0 + '/' + tempFileName
    elif checkIfThisFileExists(inputModelPathOnly+ '/' + defaultSentencePieceModelFolder1 + '/' + tempFileName) == True:
        sourceSentencePieceModel=inputModelPathOnly+ '/' + defaultSentencePieceModelFolder1 + '/' + tempFileName
    elif checkIfThisFileExists(inputModelPathOnly+ '/' + defaultSentencePieceModelFolder2 + '/' + tempFileName) == True:
        sourceSentencePieceModel=inputModelPathOnly+ '/' + defaultSentencePieceModelFolder2 + '/' + tempFileName
    elif checkIfThisFileExists(inputModelPathOnly+ '/../' + defaultSentencePieceModelFolder0 + '/' + tempFileName) == True:
        sourceSentencePieceModel=inputModelPathOnly+ '/../' + defaultSentencePieceModelFolder0 + '/' + tempFileName
    elif checkIfThisFileExists(inputModelPathOnly+ '/../' + defaultSentencePieceModelFolder1 + '/' + tempFileName) == True:
        sourceSentencePieceModel=inputModelPathOnly+ '/../' + defaultSentencePieceModelFolder1 + '/' + tempFileName
    elif checkIfThisFileExists(inputModelPathOnly+ '/../' + defaultSentencePieceModelFolder2 + '/' + tempFileName) == True:
        sourceSentencePieceModel=inputModelPathOnly+ '/../' + defaultSentencePieceModelFolder2 + '/' + tempFileName
    verifyThisFileExists(sourceSentencePieceModel,'sourceSentencePieceModel')

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
    #tempPath2=inputModelPathOnly + '/' + tempFileName
    if checkIfThisFileExists(inputModelPathOnly + '/' + tempFileName) == True:
        targetSentencePieceModel=inputModelPathOnly + '/' + tempFileName
    elif checkIfThisFileExists(inputModelPathOnly + '/../' + tempFileName) == True:
        targetSentencePieceModel=inputModelPathOnly + '/../' + tempFileName
    elif checkIfThisFileExists(inputModelPathOnly + '/' + defaultSentencePieceModelFolder0 + '/' + tempFileName) == True:
        targetSentencePieceModel=inputModelPathOnly + '/' + defaultSentencePieceModelFolder0 + '/' + tempFileName
    elif checkIfThisFileExists(inputModelPathOnly + '/' + defaultSentencePieceModelFolder1 + '/' + tempFileName) == True:
        targetSentencePieceModel=inputModelPathOnly + '/' + defaultSentencePieceModelFolder1 + '/' + tempFileName
    elif checkIfThisFileExists(inputModelPathOnly + '/' + defaultSentencePieceModelFolder2 + '/' + tempFileName) == True:
        targetSentencePieceModel=inputModelPathOnly + '/' + defaultSentencePieceModelFolder2 + '/' + tempFileName
    elif checkIfThisFileExists(inputModelPathOnly + '/../' + defaultSentencePieceModelFolder0 + '/' + tempFileName) == True:
        targetSentencePieceModel=inputModelPathOnly + '/../' + defaultSentencePieceModelFolder0 + '/' + tempFileName
    elif checkIfThisFileExists(inputModelPathOnly + '/../' + defaultSentencePieceModelFolder1 + '/' + tempFileName) == True:
        targetSentencePieceModel=inputModelPathOnly + '/../' + defaultSentencePieceModelFolder1 + '/' + tempFileName
    elif checkIfThisFileExists(inputModelPathOnly + '/../' + defaultSentencePieceModelFolder2 + '/' + tempFileName) == True:
        targetSentencePieceModel=inputModelPathOnly + '/../' + defaultSentencePieceModelFolder2 + '/' + tempFileName
    #The target is optional for fairseq, but required for ctranslate2.
    if mode == 'ctranslate2':
        verifyThisFileExists(targetSentencePieceModel,'targetSentencePieceModel')

    if __name__ == '__main__':
        print( ('Set targetSentencePieceModel to \'' + str(targetSentencePieceModel) + '\' from: \'' + targetLanguage + '\'.').encode(consoleEncoding) )


if uiPath != None:
    if checkIfThisFileExists(uiPath) == True:
        uiPath=str( pathlib.Path(uiPath).absolute() )
    else:
        print( 'Warning: Streamlit UI was specified but could not be found:\n')
        print( uiPath.encode(consoleEncoding) )
        print('')
        uiPath=None


#Update some internal variables from default values.
bpe=default_bpe
beam_size=default_beam_size
num_hypotheses=default_num_hypotheses
no_repeat_ngram_size=default_no_repeat_ngram_size
#use_vmap=default_use_vmap #Update: Added this to CLI.
inter_threads=default_inter_threads


# For best processing time with CTranslate2, CPU threads should be the same as the number of physical cores for CPU loads (not logical cores). Unclear what it should be for GPU loads but the same number as with CPU loads is a good default based upon initial testing. Update: CPU theads does not matter much when using GPU. Use default setting.
#If the user specified a number of intra_threads, as --cpuThreads, then just use that instead.
if intra_threads != None:
    pass
elif (mode=='ctranslate2') and (device=='cpu'):
    if psutilAvailable == True:
        #Always gives logical cores. Incorrect.
        #intra_threads=os.cpu_count()

        #Gives physical cores. Correct.
        intra_threads=psutil.cpu_count(logical=False)

        # Setting intra_threads=psutil.cpu_count(logical=False) always gives the wrong value for Bulldozer family FX series processors (2 Module - 4 thread ; 3 Module - 6 thread; 4 Module - 8 thread). Bulldozer FX series should use logical cores, not module count, because every logical core has some dedicated hardware to process the thread, unlike SMT.
        # https://en.wikipedia.org/wiki/List_of_AMD_FX_processors
        # Bandaid for Bulldozer FX systems on Windows.
        # This will likely hurt performance for users that have non-Bulldozer AMD FX systems. No modern AMD FX processors currently exist, so this is more of a concern for the future.
        # A proper fix might be to create alwaysUseLogicalCores.csv and look up the full processor name there, but it would be a challenge getting it fully correct due to needing the exact processor name which might require real hardware to test with which is unrealistic.
        # Alternatively, this could be exposed to the user and they could deal with it at runtime.
        # Maybe just always override this setting to whatever the user inputs? Update: Implemented this with the --cpuThreads option to allow for manual overrides.
        # This band-aid fix is currently only available on Windows.
        if sys.platform == 'win32':
            try:
                import win32com.client
                if ( str(win32com.client.GetObject('winmgmts:root\cimv2').ExecQuery('Select * from Win32_Processor')[0].Name).strip()[:6] == 'AMD FX' ):
                    intra_threads=os.cpu_count()
            except:
                pass

        # Fix for BSD systems. See:
        # https://psutil.readthedocs.io/en/latest/#psutil.cpu_count
        if intra_threads == None:
            intra_threads=default_intra_threads

    elif psutilAvailable == False:
        intra_threads=default_intra_threads
else:
    intra_threads=default_intra_threads

# Probably pointless, but just in case.
try:
    assert(isinstance(intra_threads,int))
except:
    print( 'Warning: Could not set CPU threads for CTranslate2 correctly.' )
    intra_threads=0

if ( __name__ == '__main__' ) and (verbose == True) and (mode == 'ctranslate2'):
    print ( 'CTranslate2 CPU threads=' + str(intra_threads) )


# Debug code.
#psutilAvailable=False

#Workaround to fairseq + CPU bug.
# Update: fairseq seems to hang on any sort of multiprocessing, multithreading, and even simple async + await calls.
if (mode == 'fairseq') and (device=='cpu') and (preloadModel==False) and (psutilAvailable != True):
    # Then change to preloading the model because there is no way to end the child process reliably otherwise. It hangs after it finishes processing long batches.
    preloadModel = True
    if __name__ == '__main__':
        print( '\n Warning: fairseq + CPU + multiprocessing requires psutil. Install with: \n\n    pip install psutil \n\n Since psutil is not available, preloadModel=True. \n If this behavior is not desired, install psutil.\n')
#if (mode == 'fairseq') and (device=='cpu'):
#    import signal  #Sometimes required library. This is needed to send signal.SIGTERM to terminate processes when fairseq hangs. import conditionally.


if __name__ == '__main__':
    # Print information to inform the user and help with debugging. Print it only in main since otherwise it gets printed out a lot.

    # Always print out mode (fairseq/ctranslate 2)
    print( 'mode=\''+mode + '\'' )
    # Always print out device (cpu, cuda, directml)
    print( 'device=\'' + device + '\'' )
    # Always print out source language and target language
    print( ('Source Language=\'' + sourceLanguage + '\'' ).encode(consoleEncoding) )
    print( ('Target Language= \''+ targetLanguage + '\'' ).encode(consoleEncoding) )

    if (verbose == True) or (debug == True):
    # print out model name and path
        print( ('inputModelFileNameAndPath=' + str(inputModelFileNameAndPath)).encode(consoleEncoding) )
    # print out checkpoint file name (if present, only guranteed to be valid for fairseq)
        print( ('inputModelNameWithoutPath=' + str(inputModelNameWithoutPath) ).encode(consoleEncoding) )
    # print out model path
        print( ('inputModelPathOnly=' + str(inputModelPathOnly) ).encode(consoleEncoding) )
    # print source sentencepiece_model
        print( ('sourceSentencePieceModel=' + str(sourceSentencePieceModel) ).encode(consoleEncoding) )
    # print target sentencepiece_model (only for ctranslate 2)
        print( ('targetSentencePieceModel=' + str(targetSentencePieceModel) ).encode(consoleEncoding) )

    if debug == True:
        # print out rest of variables
        print( ('preloadModel=' + str(preloadModel) ).encode(consoleEncoding) )
        print( ('perfMetrics=' + str(perfMetrics) ).encode(consoleEncoding) )
        print( ('address=' + str(address) ).encode(consoleEncoding) )
        print( ('port=' + str(port) ).encode(consoleEncoding) )
        print( ('version=' + str(version) ).encode(consoleEncoding) )
        print( ('cacheEnabled=' + str(cacheEnabled) ).encode(consoleEncoding) )
        print( ('verbose=' + str(verbose) ).encode(consoleEncoding) )
        print( ('debug=' + str(debug) ).encode(consoleEncoding) )
        print( ('tornado version=' + str(tornado.version) ).encode(consoleEncoding) )
        if mode == 'fairseq':
            print( ('fairseq version=' + str(fairseq.__version__) ).encode(consoleEncoding) )
        if mode == 'ctranslate2':
            print( ('ctranslate2 version=' + str(ctranslate2.__version__) ).encode(consoleEncoding) )
        #if device == 'directml':
            #print out directML version and torch version. Maybe OS ver as well? Since it has arbitrary requirements.


# Start app based upon input.
# fairseq will use sourceSentencePieceModel but internally.
if mode == 'fairseq':
    pass
elif mode == 'ctranslate2':
    sourceLanguageProcessor = sentencepiece.SentencePieceProcessor(sourceSentencePieceModel)
    targetLanguageProcessor = sentencepiece.SentencePieceProcessor(targetSentencePieceModel)


if preloadModel == True:
    #Then preload model.
    if mode == 'fairseq':
        # Should probably have a conditional here that says: if bpe mode == 'sentencepiece' add sentencepiece_model, else if bpe mode == pie then add ...etc    # And build the model differently based upon only the tokenizer/pbe changes since that appears to be the only condition that changes dramatically.
        # For now, add sentencepiece_model unconditionally as needed by bpe=sentencepiece, but this will need to be updated later to support additional model types.

        translator = fairseq.models.transformer.TransformerModel.from_pretrained(inputModelPathOnly,checkpoint_file=inputModelNameWithoutPath,source_lang=sourceLanguage,target_lang=targetLanguage,bpe=bpe, sentencepiece_model=sourceSentencePieceModel, no_repeat_ngram_size=no_repeat_ngram_size)

        if device == 'cuda':
            translator.cuda()
        if device == 'directml':
        # https://learn.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-windows
        # dml was defined earlier as: dml = torch_directml.device()
            translator.to(dml)

    elif mode == 'ctranslate2':
        translator = ctranslate2.Translator(inputModelPathOnly, device=device, inter_threads=inter_threads, intra_threads=intra_threads)
    else:
        sys.exit( 'Unspecified error.' )


# This still blocks because a lot of time is spent here without any pause. Maybe this should go in its own thread?
def preloadModelTranslate( rawText ):
    if mode == 'fairseq':
        return translator.translate( rawText )
    elif mode == 'ctranslate2':
        return translator.translate_batch( source=rawText , beam_size=beam_size , num_hypotheses=num_hypotheses, no_repeat_ngram_size = no_repeat_ngram_size, use_vmap=use_vmap)


async def preloadModelTranslateProxy(executor, rawText):
    return await asyncio.get_running_loop().run_in_executor(executor, preloadModelTranslate, rawText)


#def translateNMT(rawText,myQueue):
def translateNMT( rawText ):
    if debug == True:
        print( 'Processing item count: ' + str(len(rawText)) )
    if mode == 'fairseq':
        print( 'Loading fairseq in \'' + device + '\' mode for ' + str(len(rawText)) + ' entries.' )

        translator = fairseq.models.transformer.TransformerModel.from_pretrained(inputModelPathOnly,checkpoint_file=inputModelNameWithoutPath,source_lang=sourceLanguage,target_lang=targetLanguage,bpe=bpe, sentencepiece_model=sourceSentencePieceModel, no_repeat_ngram_size=no_repeat_ngram_size)

        if device == 'cuda':
            translator.cuda()
        elif device == 'directml':
        # https://learn.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-windows
        # dml was defined earlier as: dml = torch_directml.device()
            translator.to(dml)

        if (verbose == True) and (perfMetrics==True):
            startProcessingTime=time.perf_counter()

        #if device == 'cpu':
        # Process each entry individually. Does not fix bug.
        #    for textEntry in rawText:
        #        myQueue.put( translator.translate(textEntry) )

        #Batch mode. Works well.
        outputText = translator.translate(rawText)

        if (verbose == True) and (perfMetrics==True):
            processingTime=round(time.perf_counter() - startProcessingTime, 2)
            print( 'Processing time: ' + str( processingTime ) + ' seconds' )

        if debug == True:
            print(str(outputText))

        # multiprocessing.Queue logic.
        #for textEntry in outputText:
        #    myQueue.put(textEntry)

        # multiprocessing.Pipe logic.
        #myQueue.send(outputText)
        #myQueue.close()

        # concurrent.futures.ProcessPoolExecutor logic.
        # ProcessPoolExecutor is a wrapper for the multiprocessing module.
        return outputText

    elif mode == 'ctranslate2':
        print( 'Loading CTranslate2 in \'' + device + '\' mode for ' + str(len(rawText)) + ' entries.' )

        translator = ctranslate2.Translator(inputModelPathOnly, device=device, inter_threads=inter_threads, intra_threads=intra_threads)

        textAfterPreProcessing = sourceLanguageProcessor.encode(rawText, out_type=str);

        if (verbose == True) and (perfMetrics==True):
            startProcessingTime=time.perf_counter()

        outputText = translator.translate_batch( source=textAfterPreProcessing , beam_size=beam_size , num_hypotheses=num_hypotheses, no_repeat_ngram_size=no_repeat_ngram_size, use_vmap=use_vmap)

        if (verbose == True) and (perfMetrics==True):
            processingTime=round(time.perf_counter() - startProcessingTime, 2)
            print( 'Processing time: ' + str( processingTime ) + ' seconds' )

        # multiprocessing.Queue logic.
        #for i in range(len(outputText)):
        #    myQueue.put(targetLanguageProcessor.decode(outputText[i].hypotheses[0]))

        # concurrent.futures.ProcessPoolExecutor logic.
        newList=[]
        for i in range( len(outputText) ):
            newList.append( targetLanguageProcessor.decode( outputText[i].hypotheses[0] ) )
        return newList
    else:
        sys.exit( 'Unspecified error.' )


# This function allows run_in_executor() to be added to a taskList, which is a Python list, and then awaiting the taskList.
# That will process all of the entries at once with an instance of concurrent.futures.ProcessPoolExecutor .
# Otherwise, each instance of each task will block the next and also maybe the ioloop depending upon implementation details.
async def proxyTranslateNMT(executor, translateMe):
    #print( 'pie' * 200 )
    return await asyncio.get_running_loop().run_in_executor(executor, translateNMT, translateMe)


class MainHandler(tornado.web.RequestHandler):
    async def get(self):
        print('self.request=' + str(self.request) )
        if debug == True:
            print( 'Executing: ' + type(self).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_header('Content-Type', 'text/plain')
        self.set_status(200)

        self.write( 'Hello. Please use HTTP POST to communicate with ' + currentScriptNameWithoutPath)

    async def post(self):
        if perfMetrics == True:
            requestStartTime = time.perf_counter()

        self.set_header("Content-Type", 'application/json') #Set automatically by Tornado, so redundant.
        self.set_status(200)

        if debug == True:
            print('self.request=' + str(self.request) )
            client_uri = self.request.uri
            client_path = self.request.path
            client_query = self.request.query
            client_remote_ip = self.request.remote_ip
            client_url = self.request.full_url()
            print('client_uri=' + client_uri)
            print('client_path=' + client_path)
            print('client_query=' + client_query)
            print('client_remote_ip=' + client_remote_ip)
            print('client_url=' + client_url)
            print('self.get_arguments=' + str(self.get_arguments(self)))
            print('self.get_body_arguments=' + str(self.get_body_arguments(self)))

        # Basically, self.args is a dictionary made from self.request.body.
        # self.args['content'] returns all content specified in the 'content' entry.
        # if that returned item is a list, then self.args['content'][0] returns the first item in that list.
        if debug == True:
            print('self.request.body=' + str(self.request.body) )

        # Assume input is json and just blindly decode.
        #self.args = tornado.escape.json_decode(self.request.body)
        # Check if input is json, and then code. if content is not application/json, then error out.
        if self.request.headers.get('Content-Type') == 'application/json':
            self.args = tornado.escape.json_decode(self.request.body)
        else:
            print( 'Error: Only json is supported as input currently. Returning.')
            return

        if (self.args == None) or (self.args == ''):
            print( 'Error: No json contents found in request.body. Returning.')
            return
        if not isinstance(self.args,dict):
            print( 'Error: request.body did not return a Python dictionary. Returning.')
            return

        #This should print something like...
        #self.args={'content': '\xe4\xbb\x8a\xe6\x97\xa5\xe3\x82\x82', 'message': 'translate sentences'}
        #print( ('self.args=' + str(self.args)).encode(consoleEncoding) ) # Safer.
        print( 'self.args=' + str(self.args) ) # More user friendly.

        if debug == True:
            # In the json submitted via post, the 'content' entry in the dictionary should contain a single string or a python list of strings.
            if 'content' in self.args:
                print( 'content=' + str(self.args['content']) )

        if 'message' in self.args:
            if ( str(self.args['message']).lower() == 'close server' ):
                if (cacheEnabled == True) and (len(translationCacheDictionary) != 0):    
                    writeOutCache()
                print('Info: Recieved \'close server\' message. Exiting.')

                #asyncio.get_running_loop().stop()
                #asyncio.get_running_loop().stop()
                #tornado.ioloop.IOLoop.instance().stop()
                #tornado.ioloop.IOLoop.instance().stop()
                #tornado.ioloop.IOLoop.stop(self)
                #asyncio.get_running_loop().stop()
                #tornado.ioloop.IOLoop.current().add_timeout(time.time()+1, tornado.ioloop.IOLoop.current().stop())
                raise KeyboardInterrupt # Just let main() deal with this. Sloppy, but whatever.
                return
                print('This should not be printed.')

        rawInput=None
        if 'content' in self.args:
            #self.args['content'] can be a string, which is a single sentence to translate, or it can be a Python list of many strings.
            rawInput=self.args['content']
        else:
            #The data processing assumes the data is in self.args['content']. If there is another place to look, then it has to be added manually, so for now, just return if there was no 'content' entry in the submitted json.
            print( 'Error: No \'content\' entry was found in the json request.body. Returning.')
            return

        if (debug == True):
            print( ('rawInput before string conversion=' + str(rawInput)).encode(consoleEncoding) )

        convertedToList=False
        #Processing is always done using lists for compatibility with batch translations.
        #if rawInput is a string, then convert it to a list with a single entry.
        if isinstance(rawInput, list):
            pass  #Already correct.
        elif isinstance(rawInput, str):
            rawInput=[rawInput] #string convert to list
            convertedToList=True
        else:
            print( ('Error: Unrecognized type for self.args[\'content\'] body: ' + str( type(rawInput) ) ).encode(consoleEncoding) )
            return

        if verbose == True:
            print ( 'Requested number of entries=' + str(len(rawInput)) )
            #print( 'Count=' + str( len(rawInput) ) )

        if debug == True:
            print( ('rawInput after string conversion=' + str(rawInput)).encode(consoleEncoding) )
            print( 'convertedToList=' + str(convertedToList))

        if len(rawInput) == 0:
            print( 'Warning: Received empty list.' )
            return

        # Deal with cache.
        translateMe=[]
        # The syntax of this is:  tempRequestDictionary['rawEntry']=[thisValueIsFromCache,translatedData]
        #tempRequestDictionary={}
        tempRequestList=[]
        global timeCacheWasLastWritten

        if (cacheEnabled == True) and (len(translationCacheDictionary) != 0):
            # Dump rawInput into a dictionary that incorporates cache.
            # Bug: Using a dictionary creates a subtle bug where if a particular translation request has multiple duplicate items, those items will be de-duplicated.
            # That is problematic because then the len(input) will no longer match len(output). Therefore, use a python List instead to allow duplicates.
            # This does mean that duplicates will be submitted to the translation engine, but with cache enabled, this will only happen the first time.
            #create tempRequestDictionary[ 'rawEntry' ]=[ thisValueIsFromCache, translatedData ]
            #create tempRequestList.append( [ 'rawEntry', thisValueIsFromCache, translatedData ] )
            # Take every list entry from rawInput
            for i in rawInput:
                # if entryInList/translatedData exists as a key in translationCacheDictionary,
                if i in translationCacheDictionary.keys():
                    #then add entry/i to tempRequestDictionary with thisValueIsFromCache=True
                    #tempRequestDictionary[i]=[True,translationCacheDictionary[i]]
                    tempRequestList.append( [ i, True, translationCacheDictionary[i] ] )
                else:
                    # Otherwise, it needs to be processed.
                    # Create a list of all the values where thisValueIsFromCache == False. Maybe create this during parsing?
                    # Add it to the dictionary with thisValueIsFromCache=False
                    #tempRequestDictionary[i]=[False,i]
                    tempRequestList.append( [ i, False, i ] )
                    # Append it to the translateMe list.
                    translateMe.append(i)

                #Move on to next entry.
            if verbose == True:
                print( 'Number of cache hits=' + str( len(rawInput) - len(translateMe) ) )
        else:
            translateMe=rawInput

        #Then submit the translateMe list that has all rawInput without any cache hits as the list for processing. Lists are ordered.
        if debug == True:
            print( ('translateMe=' + str(translateMe)).encode(consoleEncoding) )

        postTranslatedList=[]
        #postTranslatedList.append( translateNMT( translateMe ) )
        #postTranslatedList = translateNMT( translateMe )

        # Only process if there at least one item was not found in the cache.
        if len(translateMe) != 0:
            if preloadModel == True:
                #then the models are already loaded, so just process stuff.
                print( 'Using ' + mode + ' in \'' + device + '\' mode for ' + str(len(translateMe)) + ' entries.' )
                if mode == 'fairseq':

                    if (verbose == True) and (perfMetrics==True):
                        startProcessingTime=time.perf_counter()

                    # Process each item one at a time.
                    #for textEntry in translateMe:
                    #    postTranslatedList.append( translator.translate(textEntry) )

                    # Batch processing.
                    #outputText = translator.translate(translateMe)
                    #outputText = await preloadModelTranslate(translateMe) # Still blocks.

                    # fairseq does not play well with multithreading or multiprocessing, so keep it disabled pending further troubleshooting.
                    if defaultfairseqMultithreadingEnabled == True:
                        taskList=[]
                        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                            taskList.append( asyncio.create_task( preloadModelTranslateProxy(executor, translateMe) ) )
                            # Run task directly.
                            #taskList.append(executor.submit(preloadModelTranslate, rawText)

                            #for f in asyncio.as_completed( taskList ):
                            #    outputText.append( await f )
                            outputText = await asyncio.gather( *taskList )
                            executor.shutdown(wait=False)

                        if (verbose == True) and (perfMetrics==True):
                            processingTime=round(time.perf_counter() - startProcessingTime, 2)
                            print( 'Processing time: ' + str( processingTime ) + ' seconds' )

                        #print('outputText='+str(outputText))
                        #The above returns a list which encapsulates all 1 entries in the taskList. The preloadModelTranslate function itself also returns a list, so there is a [[]] object returned.
                        #Remove the outer list.
                        outputText=outputText[0]

                        for textEntry in outputText:
                            postTranslatedList.append(textEntry)

                    elif defaultfairseqMultithreadingEnabled != True:
                        postTranslatedList = preloadModelTranslate(translateMe) 

                        #print('postTranslatedList='+str(postTranslatedList))

                        if (verbose == True) and (perfMetrics==True):
                            processingTime=round(time.perf_counter() - startProcessingTime, 2)
                            print( 'Processing time: ' + str( processingTime ) + ' seconds' )

                elif mode == 'ctranslate2':
                    textAfterPreProcessing = sourceLanguageProcessor.encode(translateMe, out_type=str);

                    if (verbose == True) and (perfMetrics==True):
                        startProcessingTime=time.perf_counter()

                    #outputText = translator.translate_batch( source=textAfterPreProcessing , beam_size=beam_size , num_hypotheses=num_hypotheses, no_repeat_ngram_size = no_repeat_ngram_size, use_vmap=use_vmap)
                    #outputText = await preloadModelTranslate(textAfterPreProcessing) #Still blocks.

                    taskList=[]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                        taskList.append( asyncio.create_task( preloadModelTranslateProxy(executor, textAfterPreProcessing) ) )

                        #for f in asyncio.as_completed( taskList ):
                        #    outputText.append( await f )
                        outputText = await asyncio.gather( *taskList )
                        executor.shutdown(wait=False)

                    if (verbose == True) and (perfMetrics==True):
                        processingTime=round(time.perf_counter() - startProcessingTime, 2)
                        print( 'Processing time: ' + str( processingTime ) + ' seconds' )

                    #The above returns a list which encapsulates all 1 entries in the taskList. The preloadModelTranslate function itself also returns a list, so there is a [[]] object returned.
                    #Remove the outer list.
                    outputText=outputText[0]

                    for i in range(len(outputText)):
                        postTranslatedList.append(targetLanguageProcessor.decode(outputText[i].hypotheses[0]))

            elif preloadModel != True:
                # if multiprocessing is allowed, then move the above core logic into a function and call that function.

                # Move data back from other process by using multiprocessing.Queue().
                #myQueue = multiprocessing.Queue()
                #translateFunction = multiprocessing.Process(target=translateNMT, args=(translateMe,myQueue,) )
                #translateFunction.start()
                # Trying to get the size of the output queue is error prone, so get the size based upon the input. This blindly assumes everything is fine.
                #for i in range(len(translateMe)):
                #    postTranslatedList.append(myQueue.get())
                #translateFunction.join()

                # multiprocessing.Pipe logic
                #localConnection, remoteConnection = multiprocessing.Pipe(False)#False means unidirectional pipe See: https://docs.python.org/3.10/library/multiprocessing.html#multiprocessing.Pipe
                #translateFunction = multiprocessing.Process(target=translateNMT, args=(translateMe,remoteConnection,) )
                #translateFunction.start()
                #postTranslatedList = localConnection.recv()
                #translateFunction.join()

                # New multiprocessing logic that should work with the I/O loop to not block the web server from functioning normally during processing. Unclear if it would be completely async and accept loading the same model a second time in a different process while the first process is still busy. That would not be a good idea. However, that is a user error, so let them deal with it.
                #https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor
                taskList = []
                with concurrent.futures.ProcessPoolExecutor( max_workers=1, mp_context = multiprocessing.get_context( defaultProcessesSpawnTechnique ) ) as executor:
                    # Add an arbitrary amount of tasks that should be completed in seperate processes to a random list.
                    # The max_workers parameter in ProcessPoolExecutor controls the number of processes to run at once.
                    #for i in range(200):
                    taskList.append( asyncio.create_task( proxyTranslateNMT(executor, translateMe) ) )

                    # Execute those processes while still in the loop so that executor still exists.
                    #for f in asyncio.as_completed( taskList ):
                    #    finalResults.append( await f )

                    # Alternative smaller and more confusing code to get finalResults. Removing the asterisk * breaks it.
                    # https://stackoverflow.com/questions/36901/what-does-double-star-asterisk-and-star-asterisk-do-for-parameters
                    #finalResults = await asyncio.gather( *taskList )
                    postTranslatedList = await asyncio.gather( *taskList ) # *postTranslatedList supposedly means 'unpack postTranslatedList' which still does not clarify its usage. Why does the assignment break when removing it? Maybe it is not a return object, but the actual stored create_task functions themselves? But in that case, then should not just feeding the raw taskList also work without unpacking? What does asyncio.gather() expect?
                    #Hint: https://docs.python.org/3/library/asyncio-subprocess.html#subprocesses

                    maxBatchSizeForFairseqBug=5  # Magic number.
                    # Sizes of ~25+ on fairseq CPU multiprocess always produce a bug on tested CPU that causes the subprocess to hang and never return once its calculations are complete. Does not occur in the same exact multiprocess code if cuda is enabled, or in --preloadModel mode CPU using same code. Does not occur with ctranslate2. multiprocessing.Queue vs multiprocessing.Pipe logic does not matter. Might be an internal bug in fairseq that is somehow triggered by multiprocessing but only sometimes?
                    # 20 does not usually produce fairseq cpu multiprocess hang bug, but might depend on CPU or utilization %, flat time, or other unknown factors. Smaller sizes are less likely to produce this intermittent bug. Bug was reproduced at least 1 time at batch size=10. Bug has not been reproduced yet at batch size <= 5.
                    # New improved workaround for this bug is just to forcequit the process after telling executor to shut down.
                    if ( mode == 'fairseq' ) and ( device == 'cpu' ) and ( len(translateMe) > maxBatchSizeForFairseqBug ):
                        executor.shutdown(wait=False,cancel_futures=True) #This actually makes the process return the results reliably, but there is no way to shut down the child process without knowing it's Process ID which ProcessPoolExecutor does not expose aparently? The multiprocessing module does expose this information, so there might be some workaround. https://docs.python.org/3/library/multiprocessing.html#the-process-class
                        # One alternative could be to change the datastructure so that it always returns back its processID, but that is a lot to change for a workaround for a specific bug in one configuration.
                        # So instead use psutil to find the PID which, in turn, makes psutil a required dependency for fairseq + CPU + multiprocessing. Band-aid fix is better than no fix.
                        try:
                            for process in psutil.Process(os.getpid()).children(recursive=True):
                                # The UI convinence function might be mixed in here which spawns several processes, so it is important to only close the correct one. Blindly selecting the first or last one does not work because the entries seem to be returned out of spawning order.
                                # Docs: https://psutil.readthedocs.io/en/latest/#processes
                                # Syntax: psutil.Process().cmdline()
                                if debug == True:
                                    print( 'process.pid()=' + str(process.pid) )
                                    print( 'process.name()=' + str(process.name()) )
                                    print( 'process.exe()=' + str(process.exe()) )
                                    print( 'process.cmdline()=' + str(process.cmdline()) )
                                # psutil.Process().cmdline() returns a list of strings, so check the list to see if it was spawned using Python's multiprocessing module to identify the correct one.
                                for i in process.cmdline():
                                    if i == '--multiprocessing-fork': #This is used for spawn as well.
                                        #process.send_signal(signal.SIGTERM)
                                        process.terminate() #Mostly an alias for above code.
                            if debug == True:
                                print('Info: Child processes found and sent signal.SIGTERM.')
                        except psutil.NoSuchProcess:
                            if debug == True:
                                print('No such child process.')
                    else:
                        executor.shutdown(wait=False)

                #The above returns a list which encapsulates all 1 entries in the taskList. translateNMT itself also returns a list, so there is a [[]] object returned.
                #Remove the outer list.
                postTranslatedList=postTranslatedList[0]


        if debug == True:
            print( ( 'postTranslatedList=' + str(postTranslatedList) ).encode(consoleEncoding) )
            if cacheEnabled == True:
                print( 'translationCacheDictionary length=' + str(len( translationCacheDictionary )) )

        # Initalize finalOutputList
        finalOutputList=[]
        if cacheEnabled == True:
            # Decide!
            # Need to merge processed values with cache hits.
            # Initalize a dumbCounter=0
            counter=0
            # if literally every single request value ended up being found in the cache and processing was skipped,
            # then set the finalOutputList to the values in the dictionary.
            # Alternatively, if the cache was just initalized and there were no entries in the cache before processing, then do not attempt to merge an empty cache with processed items.
            # The translations are stored in tempRequestDictionary as:
            # tempRequestDictionary['rawEntry']=[thisValueIsFromCache,translatedData]
            # tempRequestList.append( [ 'rawEntry', thisValueIsFromCache, translatedData ] )
            if len(postTranslatedList) == 0:
                #for i in tempRequestDictionary.values():
                for i in tempRequestList:
                    finalOutputList.append(i[2])
            elif len(translationCacheDictionary) == 0:
                finalOutputList=postTranslatedList
            else:
                # Need to merge processed items with dictionary for final output.
                # On return from processing, iterate over the tempRequestDictionary. For every entry.
                #for key, value in tempRequestDictionary.items():
                for i in tempRequestList:
                    # if thisValueIsFromCache == True:
                    if i[1] == True:
                        # Then add to finalOutputList as-is and move to the next entry in the dictionary.
                        finalOutputList.append( value[2] )
                    # if thisValueIsFromCache == False:
                    elif value[1] == False:
                        #Then obtain the value to add to finalOutputList from postTranslatedList, the list that has the translated values,
                        # and add that translated entry[counter] to the final output list
                        finalOutputList.append(postTranslatedList[counter])
                        # increment the counter and go to the next entry in the dictionary
                        counter += 1
                    else:
                        sys.exit( 'Unspecified error')

            if len(postTranslatedList) != 0:
                # Add all newlyTranslated entries found to translationCacheDictionary.
                counter=0
                # for entry in postTranslatedList
                # for every untranslated entry, update the cache with the untranslated entry and the translated line together as a pair.
                # Wait, is this logic correct? translateMe is the list right before it gets submited for translation. postTranslatedList is the post-translated list.
                # As long as both lists are exactly the same length and no errors occured, then this will work. Should that be asserted or double checked somehow?
                # The issue being that it is difficult to understand what to do if they do not match, except to print the mismatch to the screen. Since that is incredibly cryptic to explain, just let the program crash instead.
                for entry in translateMe:
                    translationCacheDictionary[translateMe[counter]] = postTranslatedList[counter]
                    counter += 1

        # if cacheEnabled != True:
        else:
            finalOutputList=postTranslatedList

        # if the input was originally a string, then convert it back to a string for output.
        if convertedToList == True:
            finalOutputList=finalOutputList[0]

        #if (verbose == True) or (debug == True):
        #    print(str(finalOutputList).encode(consoleEncoding))
        print( str(finalOutputList) )

        if cacheEnabled == True:
            #Check timer for cache last written. If timer > 60s, then write out to file.
            if int( time.perf_counter()  - timeCacheWasLastWritten) > defaultSaveCacheInterval:
                timeCacheWasLastWritten=time.perf_counter()
                try:
                    writeOutCache()
                except:
                    print( 'Warning: An unspecified error occured when writeOutCache.')

        if perfMetrics == True:
            #requestServicingTime=round( time.perf_counter()  - requestStartTime, 2)

            # Referencing the processingTime here is sort of pointless if the model is preloaded because the processingTime and requestServicingTime will be very similar. However, if the model is not preloaded, then it cannot be referenced directly because it is part of another process. It would have to be handed back using a pipe or queue, but that would be complicated for very little benefit, so just settle for printing it out to the screen from the other process.
            # The user can figure out the model loading time on their own if they want.
            #if preloadModel != True:  
            #    print( 'Model loading time: ' + str ( round( requestServicingTime - processingTime, 2) ) + 's' )

            #print( 'Request servicing time: ' + str( requestServicingTime )+ 's')
            print( 'Request servicing time: ' + str( round( time.perf_counter()  - requestStartTime, 2) )+ 's')

        #return self.write(json.dumps(finalOutputList))
        self.write( json.dumps(finalOutputList) )


# At some point, this should be hardened.
# Documentation:
# https://www.tornadoweb.org/en/stable/web.html
# https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers
# https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Type
# https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types
# set output response header:
# self.set_header('Content-Type', 'application/json')
# self.set_header('Content-Type', 'application/pdf')
# self.set_header('Content-Type', 'application/zip')
# self.set_header('Content-Type', 'application/x-rar-compressed')
# self.set_header('Content-Type', 'application/octet-stream')
# self.set_header('Content-Type', 'audio/mpeg')
# self.set_header('Content-Type', 'image/jpeg')
# self.set_header('Content-Type', 'image/png')
# self.set_header('Content-Type', 'text/plain')
# self.set_header('Content-Type', 'text/html')
# self.set_header('Content-Type', 'text/css')
# self.set_header('Content-Type', 'text/javascript')
# self.set_header('Content-Type', 'text/csv')
# self.set_header('Content-Type', 'video/mp4')

# self.set_header('Server', 'tornado/'+str(tornado.version)) #This is incorrect. Tornado automatically sets this correctly on its own. Example response headers for a 404:
# HTTP/1.1 404 Not Found
# Server: TornadoServer/6.4
# Content-Type: text/html; charset=UTF-8
# Date: Sun, 01 Feb 2020 12:00:00 GMT
# Content-Length: 69


class ReturnVersion(tornado.web.RequestHandler):
    async def get(self):
        print('self.request=' + str(self.request) )
        if debug == True:
            print( 'Executing: ' + type(self).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status(200)
        self.set_header('Content-Type', 'text/plain')

        self.write( scriptNameWithVersion )

    async def post(self):
        print('self.request=' + str(self.request) )
        if debug == True:
            print( 'Executing: ' + type(self).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status(200)
        self.set_header('Content-Type', 'application/json')

        self.write( json.dumps( scriptNameWithVersionDictionary ) )


class ReturnModel(tornado.web.RequestHandler):
    async def get(self):
        print('self.request=' + str(self.request) )
        if debug == True:
            print( 'Executing: ' + type(self).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status(200)
        self.set_header('Content-Type', 'text/plain')

        self.write( modeAndModelName )

    async def post(self):
        print('self.request=' + str(self.request) )
        if debug == True:
            print( 'Executing: ' + type(self).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status(200)
        self.set_header('Content-Type', 'application/json')

        self.write( json.dumps( modeAndModelNameDictionary ) )


class SaveCache(tornado.web.RequestHandler):
    async def get(self):
        print( 'self.request=' + str(self.request) )
        if debug == True:
            print( 'Executing: ' + type(self).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status(200)
        self.set_header('Content-Type', 'text/plain')

        if cacheEnabled != True:
            self.finish( 'Unable to save cache because cache is not enabled.' )
            return

        global timeCacheWasLastWritten
        #Check timer for cache last written. If timer > 60s, then write out to file.
        if int( time.perf_counter()  - timeCacheWasLastWritten) > defaultSaveCacheInterval:
            timeCacheWasLastWritten=time.perf_counter()
            try:
                writeOutCache()
                self.finish('Cache was written to disk.')
                return
            except:
                print( 'Warning: An unspecified error occured during writeOutCache()' ) # Print to console.
                self.finish( 'Warning: An unspecified error occured during writeOutCache()' ) # Send error message over HTTP.
                return
        else:
            self.finish('Cache was not written to disk. To save cache, please wait up to ' + str(defaultSaveCacheInterval) + ' seconds.')
            return

    async def post(self):
        print( 'self.request=' + str(self.request) )
        if debug == True:
            print( 'Executing: ' + type(self).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status(200)
        self.set_header('Content-Type', 'application/json')

        if cacheEnabled != True:
            self.finish( json.dumps({ 'content': 'Unable to save cache because cache is not enabled.'}) )
            return

        global timeCacheWasLastWritten
        #Check timer for cache last written. If timer > 60s, then write out to file.
        if int( time.perf_counter()  - timeCacheWasLastWritten) > defaultSaveCacheInterval:
            timeCacheWasLastWritten=time.perf_counter()
            try:
                writeOutCache()
                self.finish( json.dumps({'content': 'Cache was written to disk.'}) )
                return
            except:
                print( 'Warning: An unspecified error occured during writeOutCache()' ) # Print to console.
                self.finish( json.dumps({'content': 'Warning: An unspecified error occured during writeOutCache()'}) ) # Send error message over HTTP.
                return
        else:
            self.finish( json.dumps({'content': 'Cache was not written to disk. To save cache, please wait up to ' + str(defaultSaveCacheInterval) + ' seconds.'}) )
            return


class ClearCache(tornado.web.RequestHandler):
    async def get(self):
        print( 'self.request=' + str(self.request) )
        if debug == True:
            print( 'Executing: ' + type(self).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status(200)
        self.set_header('Content-Type', 'text/plain')

        if cacheEnabled != True:
            self.finish( 'Unable to clear cache because cache is not enabled.' )
            return

        global timeCacheWasLastCleared
        #Check timer for cache last written. If timer > 60s, then write out to file.
        if int( time.perf_counter()  - timeCacheWasLastCleared) > defaultMinimumClearCacheInterval:
            timeCacheWasLastCleared=time.perf_counter()
            clearCache()
            self.finish( 'Cache was cleared.' )
            return
        else:
            self.finish( 'Cache was not cleared. To clear cache, please wait up to ' + str(defaultMinimumClearCacheInterval) + ' seconds.' )
            return

    async def post(self):
        print( 'self.request=' + str(self.request) )
        if debug == True:
            print( 'Executing: ' + type(self).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status(200)
        self.set_header('Content-Type', 'application/json')

        if cacheEnabled != True:
            self.finish( json.dumps({'content': 'Unable to clear cache because cache is not enabled.'}) )
            return

        global timeCacheWasLastCleared
        #Check timer for cache last written. If timer > 60s, then write out to file.
        if int( time.perf_counter()  - timeCacheWasLastCleared) > defaultMinimumClearCacheInterval:
            timeCacheWasLastCleared=time.perf_counter()
            clearCache()
            self.finish( json.dumps({'content':'Cache was cleared.'}) )
            return
        else:
            self.finish( json.dumps({'content':'Cache was not cleared. To clear cache, please wait up to ' + str(defaultMinimumClearCacheInterval) + ' seconds.' }) )
            return


class GetCache(tornado.web.RequestHandler):
    async def get(self):
        print( 'self.request=' + str(self.request) )
        if debug == True:
            print( 'Executing: ' + type(self).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status(200)

        if cacheEnabled != True:
            self.set_header('Content-Type', 'application/json')
            self.finish( json.dumps({'content': 'Unable to send cache because cache is not enabled.'}) )
            return

        #https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Disposition
        self.set_header('Content-Type', 'application/csv')
        self.set_header('Content-Disposition', 'attachment; filename=' + cacheFileNameOnly )

        # This might produce an error if the file has not been written to disk yet.
        # It might be better to read the entire file into memory, as cumbersome as that is, and then send it. That minimizes the potential of writing to the file at the same time as reading it. That wastes a lot of memory that will never be reclaimed by the OS, even if del is explcitly called on the object, however. So, which is better? Which is worse? Oh, the joys of async programming.
        chunkSize = 4194304 #4MB
        with open(cacheFilePathAndName, 'rb') as myFileHandle:
            while True:
                chunk = myFileHandle.read(chunkSize)
                if not chunk:
                    break
                try:
                    self.write(chunk)
                    await self.flush()
                    await asyncio.sleep(0)
                except:
                    break
                finally:
                    del chunk

    async def post(self):
        print( 'self.request=' + str(self.request) )
        if debug == True:
            print( 'Executing: ' + type(self).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status(200)
        self.set_header('Content-Type', 'application/json')

        if cacheEnabled != True:
            self.finish( json.dumps({'content': 'Unable to send cache because cache is not enabled.'}) )
            return

        self.finish( json.dumps( dict( [('content',translationCacheDictionary)] ), ensure_ascii=False) )
        return


async def runUI(uiPath):
    # Might be useful somehow: https://docs.python.org/3.8/library/shlex.html#shlex.quote
    #import subprocess
#    myPath=os.path.join(currentScriptPathOnly,str(pathlib.Path(os.path.join( 'resources', 'webUI.py'))))

    myString='\" --server.address='+address+' -- ' + '--address ' + address + ' --port ' + str(port) + ' --quiet'
    #myString=''

    # This syntax starts a fully independent instance of the UI. Great for stability, but not for managing the subprocess.
    # It works, but stars a new shell Window. Makes it more obvious that it needs to cleanly shut down at least.
#    if sys.platform == 'win32':
#        fullCommand='start "py3translationServer UI - by gdiaz384" streamlit run ' + myPath + myString
#    elif sys.platform == 'linux':
#        fullCommand='bash -c streamlit run ' + myPath + myString
    # This syntax is preferred by subprocess.run()
    #fullCommand='streamlit.exe run \"' + uiPath + myString
    # This should be cross platform but might require a global streamlit install.
    fullCommand='streamlit run \"' + uiPath + myString
    # This syntax of invoking streamlit as a python module has the same problem as above.
    #fullCommand='python -m streamlit run \"' + uiPath + myString

    # This UI launching syntax might not play well with portable versions of Python.
    # Need to test. If it does not work, then find a way to locate the currently running Python.exe
    # and then always launch as a module with that dynamically located python environment.
    # For compiled versions of py3translationServer.exe, just do not support launching the UI that way maybe?
    # Could also check the extension and launch stuff differently.
    # Streamlit itself is Apache (license) + Python. Could just fully integrate the code base to avoid having a seprate process for the UI?

    if debug==True:
#        print( 'myPath=' + myPath )
        print( 'uiPath=' + uiPath )
        print( 'myString=' + myString )
        print( 'fullCommand=' + fullCommand)

    # Without the fully independent process fix, start "" on Windows and bash -c on Linux, this does not work.
    # The programs launch but there is some sort of conflict and they do not work either together.
    # Update: The subprocess launches correctly, but cannot communicate with the server.
    # if the subprocess is forcequit, then py3translationServer starts working again.
    # In other words, it seems like starting the subprocess locks out the main process until the subprocess completes.
    #subprocess.run(fullCommand, capture_output=False, shell=True)
    #subprocess.run(fullCommand)
    
    # asyncio + exec Does not work.
    #await asyncio.create_subprocess_exec(fullCommand)

    # This works perfectly. So, the solution was to use asyncio with a shell.
    # https://docs.python.org/3/library/asyncio.html
    # https://docs.python.org/3/library/asyncio-subprocess.html
    # https://docs.python.org/3.8/library/asyncio-subprocess.html#asyncio.asyncio.subprocess.Process
    try:
        return await asyncio.create_subprocess_shell(fullCommand)
    except:
        return None

    # This works regardless of syntax, but does not cleanly shut down.
    # It seems to still respond in the existing shell unless using the independent shell syntax, so maybe there is a way to make it work?
#    uiHandle = subprocess.Popen(fullCommand, shell=True)#,stdout='PIPE', stderr='PIPE'
#    print('uiHandle.pid='+str(uiHandle.pid))
#    time.sleep(5)
    #print('pie')
    #uiHandle.kill()
    #import signal
    #uiHandle.send_signal(signal.SIGTERM)
    #uiHandle.terminate()
#    return
#    if sys.platform == 'win32':
#        print('pie')
#        os.kill(uiHandle.pid, signal.SIGTERM)
#    else:
#        print('pie2')
#        os.kill(uiHandle.pid, signal.SIGTERM)

# Investigation results: streamlit.exe needs to be run from a shell, but does not explicitly require it after initalization. 
# streamlit itself starts a python subprocess which then executes script.py in: streamlit run script.py
# This creates a process chain that looks like: python3translationServer -> shell -> streamlit (semi-native executable) -> script.py
# The above code does not work because it only closes the shell, which the streamlit executable does not need to run.
# On Windows, manually closing cmd.exe will cause it to enforce a 5 second hard-timeout on subprocesses,
# but this timeout does not apply if closing the command prompt programatically.

"""
    # Only psutil works as intended because it can recursively identify the subprocesses.
    try:
        for process in psutil.Process(os.getpid()).children(recursive=True):
            #process.send_signal(signal.SIGTERM)
            process.terminate() #Mostly an alias for above code.
        if verbose == True:
            print('Info: Child processes found and sent signal.SIGTERM.')
    except psutil.NoSuchProcess:
        if verbose == True:
            print('No child processes.')
"""

# Possible workarounds to avoid creating zombie processes:
# 1) streamlit.cmd has an alternative syntax for launching streamlit: python -m streamlit %*
# TODO: Check what the process chain looks like when using that syntax, but it almost certainly looks like python3translationServer -> streamlit (python) -> script.py (python)
# This might work if the script.py (python) process always closes when streamlit.exe closes. This seems promising. Need to test.
# 2) Always use psutil to close all subprocesses, but this requires psutil. Would have to disable ui launching functionality if psutil is not installed.
# 3) Use alternative syntax for identifying and closing subprocesses. What syntax is that...? Platform specific stack tracing seems likely.
# 4) compile streamlit to a true native executable. This might work if the script.py (python) process always closes when streamlit.exe closes.
# Limitations: requires compiling streamlit, platform specific


async def main():

#    Define v0 API
#    application = tornado.web.Application([
#            (r'/version', ReturnVersion),
#            (r'/api/v1/version', ReturnVersion),
#            (r'/model', ReturnModel),
#            (r'/api/v1/model', ReturnModel),
#            (r'/', MainHandler),
#            ])

    #Define v1 API
    translationAPIv1=[
        (r'/', MainHandler),
        (r'/version', ReturnVersion),
        (r'/api/v1/version', ReturnVersion),
        (r'/model', ReturnModel),
        (r'/api/v1/model', ReturnModel),
        (r'/api/v1/saveCache', SaveCache),
        (r'/api/v1/writeCache', SaveCache),
        (r'/api/v1/clearCache', ClearCache),
        (r'/api/v1/getCache', GetCache),
        ]

    # Make application that uses the above API. Application can bind to localhost (with IP alias), all addreses, or a specific address.
    # Requiring HostMatches(address) means that DNS rebind attacks will not work.
    # https://www.tornadoweb.org/en/stable/guide/security.html#dnsrebinding
    if (address == 'localhost') or (address == '127.0.0.1'):
        application = tornado.web.Application([ (tornado.web.HostMatches( r'(localhost|127\.0\.0\.1)' ), translationAPIv1 ), ])
    elif (address == '0.0.0.0'):
        application = tornado.web.Application( translationAPIv1 )
    else:
        application = tornado.web.Application([ (tornado.web.HostMatches( address ), translationAPIv1 ), ])

    print( (currentScriptNameWithoutPath + ' v' + __version__).encode(consoleEncoding) )
    print( (currentScriptNameWithoutPath + ' ' + mode + ' ' + device + ' started: http://' + str(address) + ':' + str(port) ).encode(consoleEncoding) )
    # if binding to all addresses, then display the connectable addresses for convenience.
    if ( address == '0.0.0.0' ):
        print( 'http://localhost:' + str(port) )
        import socket
        for i in socket.getaddrinfo(socket.gethostname(),None):
            #print( 'http://' + str(i[4][0]) + ':' + str(port) )
            temp=str(i[4][0])
            # filter out IPv6 addresses
            if temp.find(':') == -1:
                print( 'http://' + temp + ':' + str(port) )

    # Update this with: https://www.tornadoweb.org/en/stable/netutil.html Done.
    application.listen(address=address, port=port)

    global uiHandle
    uiHandle=None
    if uiPath != None:
        uiHandle = await runUI(uiPath)
    # uiHandle is an instance of: asyncio.subprocess.Process
    #https://docs.python.org/3.8/library/asyncio-subprocess.html#interacting-with-subprocesses

    await asyncio.Event().wait()

if __name__ == '__main__':

    if perfMetrics == True:
        print( 'Load time: ' + str( round(time.perf_counter() - startedLoadingTime, 2) ) + ' seconds' )

#    try:
    try:
        asyncio.run( main() )
    except KeyboardInterrupt:
        tornado.ioloop.IOLoop.instance().stop()
        #asyncio.get_running_loop().stop()
#    except RuntimeError:
#        pass


    if psutilAvailable == True:
        #Only psutil works as intended to close the UI.
        try:
            for process in psutil.Process(os.getpid()).children(recursive=True):
                #process.send_signal(signal.SIGTERM)
                process.terminate() #Mostly an alias for above code.
            if verbose == True:
                print('Info: Child processes found and sent signal.SIGTERM.')
        except psutil.NoSuchProcess:
            if verbose == True:
                print('No child processeses.')
    else:
        # This must be below the psutil code that closes subprocesses or the linking process will not exist for psutil to use. See:
        # https://psutil.readthedocs.io/en/latest/#psutil.Process.children
        if (uiPath != None) and (uiHandle != None):
            # This does not work because it only closes the shell instance, not the streamlit + pythonScript.py subprocess.
            #uiHandle.send_signal(signal.SIGTERM)
            uiHandle.terminate() #This is an alias for the above command but with cross platform support.


    if (mode == 'fairseq') and (device == 'cpu'):
        #print('pie',flush=True)
        psutilAvailable=False
        if psutilAvailable == True:
            #print('pie2', flush=True)
            psutil.Process(os.getpid()).send_signal(signal.SIGTERM) # Suicide. The safer way.
        elif psutilAvailable != True:
            #print('pie3', flush=True)
            os.kill(os.getpid(),signal.SIGTERM) # Suicide.

    sys.exit('Program crashed successfully.')

    # Might be useful: https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Process.terminate
    # This seems to only be useful when constructing the process manually. Might still be useful if the multiprocessing code is updated to remove the psutil requirement, but that does not remove the psutil requirement from the HTTP API's shutdown command that is responsible for doing the same thing. In other words, meh. Just do nothing instead. TODO: Double check if this is still true.

