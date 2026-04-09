#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Description:
py3translationServer.py exposes fairseq and CTranslate2 models over HTTP using the Tornado web server.

- Tornado is a Python web framework and asynchronous networking library with an emphasis on non-blocking network I/O.
- fairseq is library for machine learning and data modeling.
- CTranslate2 is a C++ and Python library for efficient inference with transformer models, including those used by fairseq.
- "Transformers is a library of pretrained natural language processing, computer vision, audio, and multimodal models for inference and training."
- More information:
    - https://www.tornadoweb.org
    - https://github.com/facebookresearch/fairseq
    - https://opennmt.net/CTranslate2
    - https://huggingface.co/docs/transformers/main/en/index

Install with:
- pip install tornado ctranslate2 transformers
- fairseq must be built from source since the 0.2.0 version available on PyPi is too old.

py3translationServer.py features:
- Supports both CPU and GPU inferencing. 'GPU' is aliased to CUDA, but DirectML is also supported on Windows.
- Supports large batch requests.
- Supports both single process and multiprocess modes.
    - In single process mode, the model is preloaded for low latency inferencing.
    - In multiprocess mode, the model has significantly longer initial startup time (5 seconds+) but returns all memory allocated once the transaction completes. This is ideal for batch translations and long term operation.

Copyright: github/gdiaz384
License: AGPLv3, https://www.gnu.org/licenses/agpl-3.0.html
"""
__version__ = '2025.06.06'


def getDefaults():
    # Set global defaults:
    defaults = { }
    defaults[ 'modelDatabase' ] = [ 'fairseq.sugoi', 'ctranslate2.sugoi', 'transformers.marianmt' ]
    # The [ 'modelDatabase' ] list should be turned into a modelDatabase={ } dictionary where each key is a model name mapping to a value. Then, the value should be another dictionary that has these values.
    # [ 'imported' ]=False, [ 'module' ]=importlib.import_module( 'fully.qualified.path.to.module' ), [ 'model' ]=[ 'module' ].Translator(), [ 'available' ]=False, [ 'hash' ]=None,
#>>> string=[ 'fairseq-sugoi' ]
#>>> sugoi_module = importlib.import_module('resources.engines.'+string[ 0 ].split( '-' )[ 0 ]+'.'+ string[ 0 ].split( '-' )[ 1 ])
#>>> sugoi_module.Translator()

    # Set main program defaults:
    # Valid values: None, cpu, gpu, cuda, directml. gpu is aliased to cuda.
    # ROCm support is not currently implemented. Entering 'rocm' will use fairseq in CPU mode and error out CTranslate2.
    defaults[ 'device' ] = None
    # Host address and port. 0.0.0.0 means 'bind to all local addresses'.
    # localhost, 127.0.0.1, 0.0.0.0
    defaults[ 'address' ] = 'localhost'  # localhost has an alias of 127.0.0.1
    defaults[ 'port' ] = 14366

    # Load a model into memory immediately. True, (False)
    defaults[ 'preloadModel' ] = False
    # If preloadModel is True, must be fairseq, ctranslate2, or transformers.
    defaults[ 'engine' ] = None
    # If preloadModel is True, this is the name of the model to preload.
    defaults[ 'modelName' ] = None
    # If preloadModel is True, the path to model.bin.
    defaults[ 'modelPath' ] = None
    # If preloadModel is True, the initial source and target languages. Different models use different language code formatting.
    defaults[ 'sourceLanguage' ] = None
    defaults[ 'targetLanguage' ] = None

    # Enable caching of translations. (True), False
    defaults[ 'cacheEnabled' ] = True
    # Set to False to overwrite cache.csv in-place without creating a copy. (True), False. Currently unused.
    #defaults[ 'createBackupOfCacheFile' ] = True
    # cachePath is normally used to store cache. Setting storeCacheInLocalEnvironment True changes the storage location of the cache to:
        # Windows: os.getenv( 'LOCALAPPDATA' )/py3translationServer/cache
        # Linux: str( pathlib.Path('~').expanduser() )+/.cache/py3translationServer/cache
    defaults[ 'storeCacheInLocalEnvironment' ] = False
    # Path to the folder that will be used to store cache. Can be relative to program.py/.exe or absolute. Only used if storeCacheInLocalEnvironment is False.
    defaults[ 'cachePath' ] = 'resources/cache'
    # The minimum number of seconds that must pass before the next request will trigger writing the cache to disk. Set to low value, like 1 to nearly always write out file.
    # In some situations, writing the file may take several seconds. The lowest safe amount should be ~10 assuming a healthy disk and low to moderate active network i/o. Set to higher amounts to minimize disk i/o of mostly redundant data.
    defaults[ 'saveCacheInterval' ] = 60
    # The minumum time to wait in between allowing cache to be cleared meaning that cache cannot be cleared within this window of writing it out.
    # Not implemented yet.
    defaults[ 'clearCacheInterval' ] = 60

    # Valid values are spawn, fork, and forkserver. Changing this will result in untested behavior.
    # https://docs.python.org/3.12/library/multiprocessing.html#contexts-and-start-methods
    defaults[ 'processesSpawnTechnique' ] = 'spawn'
    # fairseq does not play well with multithreading or multiprocessing, so create a toggle to help troubleshooting.
    defaults[ 'fairseqMultithreadingEnabled' ] = True

    defaults[ 'fileEncoding' ] = 'utf-8'
    defaults[ 'consoleEncoding' ] = 'utf-8'
    #https://docs.python.org/3.8/library/codecs.html#error-handlers
    defaults[ 'inputFileErrorHandling' ] = 'strict'
    if sys.version_info.minor >= 5
        defaults[ 'outputFileErrorHandling' ] = 'namereplace'
    else:
        defaults[ 'outputFileErrorHandling' ] = 'backslashreplace'

    return defaults


#Might be an interesting read: https://docs.python.org/3/library/configparser.html
import argparse                # Used to add command line options.
import sys                        # End program on fail condition. Technically, this always exits as an error for anything but sys.exit(0) even if just trying to close normally, but w/e.
import os                          # Test if file/folder exists.
import platform                 # Used to test for Windows, Linux to implement platform specific code and thus iterlopability.
#import io, iostream, gen  # Optional. Read from and write to objects in memory as if they were files. Used for sending cache.csv directly from memory and perhaps will be used later for cache.csv.zip. Not fully implemented yet. Import conditionally later if needed.
#import socket                   # Optional. Used to get IP's and print them to clarify to the user where Tornado is listening. Import as needed.
import pathlib                   # Part of standard library since 3.4. Imported for Path class which does sane path handling.
import json                       # Accept JSON as input. Return JSON after processing.
import time                      # Optional library. Used to calculate performance metrics. Import conditionally later. #Update: perfMetrics, cache write out time, and clear cache time require this, so just always include it instead. Part of standard library anyway.
#import csv                        # Used to read/write cache files. Import conditionally later based upon if cache is enabled or not. Update: Part of multiLanguageCache library now.
#import date or datetime   # Humm. Could be used to append the current date to the cache backup file as cache.hash.csv.backup.Today.csv
import signal                   #Sometimes required library. This is needed to send signal.SIGTERM to terminate processes when fairseq + CPU hangs. import conditionally as needed. Also used for UI.
#import inspect               #Used to print out the name of the current function during execution which is useful when debugging. Import conditionally later.
#import hashlib                 # Used to identify correct cache.csv on disk and also as a psudo-rng function for temporary writes. Update: Part of multiLanguageCache library now.
import configparser         # Used to read/write .ini files. https://docs.python.org/3/library/configparser.html

#import fairseq                 # Core engine. Must be installed with 'pip install fairseq' or built from source. Import conditionally later.
#import ctranslate2           # Core engine. Must be installed with 'pip install ctranslate2'. Import conditionally later.
#import sentencepiece      # Core dependency. Must be installed with 'pip install sentencepiece' Used for both fairseq and ctranslate2. However, fairseq will import it internally, like with PyTorch, so do not worry about it explicitly unless ctranslate2 is specified.

import asyncio                # Used for asynconous I/O. Part of standard library since 3.4. Is also a tornado dependency.
import multiprocessing     # Part of standard library. Used for Process, Queue objects. Used in core logic and also in cache logic. #Should probably import conditionally. #Update, this is still needed, even with concurrent.futures, to deterministically set the spawn type for the child processes, spawn, fork, forserver, but is still technically optional if cache is not enabled and if preloadModel==True. Annoying to import conditionally. Also used for multiprocessing.freeze_support()
import concurrent.futures # Used to create a process that can work with asynconous I/O. Basically asyncio + multiprocessing.
import tornado                 # Web server. tornado.escape.json_decode creates Python dictionary from input json. Must be installed with 'pip install tornado'.
import tornado.web          # This duplicate explicit import improves compatibility with Python versions < 3.8 and pyinstaller.
try:
    global psutilAvailable
    import psutil                 # This library is required for fairseq + CPU + multiprocessing, but technically optional otherwise. This library is also used to optimize CTranslate2 to use the number of physical cores if running on CPU. #Update: It should be possible to remove this requirement by altering the way the new process returns its data to always return the process ID. However, the signal library would still be required and sending signal.SIGTERM to the process might be more complicated, os specific, or unsafe. Update: This is also used to identify and child processes when launching the UI in order to close them during shutdown, so back in required territory.
    psutilAvailable=True
except ImportError:
    psutilAvailable=False
try:
    import resources.commonFunctions as commonFunctions
except:
    sys.path.append( str( pathlib.Path( __file__ ).resolve().parent ) + '/resources' )
    #print(sys.path)
    import commonFunctions




def createCommandLineOptions( defaults, usageHelp ):
    commandLineParser = argparse.ArgumentParser( description = 'Description: '+ pathlib.Path( __file__ ).name + ' exposes NMT models over HTTP using the Tornado web server with a Sugoi API. Supports fairseq, CTranslate2, and transformers engines. ' + usageHelp)
    commandLineParser.add_argument( '-dev', '--device', help = 'Process using cpu, gpu, cuda, or directml. gpu is aliased to cuda. rocm is not supported yet. Default = '+defaultDevice, default = defaultDevice, type = str )
    commandLineParser.add_argument( '-a', '--address', help = 'Specify the address to listen on. To bind to all addresses, use 0.0.0.0  Default is to bind to: '+ str( defaultAddress ), default = defaultAddress, type = str)
    commandLineParser.add_argument( '-port', '--port', help = 'Specify the port the local server will use. Default = ' + str(defaultPort), default = defaultPort, type = int )

    # preloadModel settings
    commandLineParser.add_argument( '-pm', '--preloadModel', help = 'Load the model into memory immediately and make the system run out of memory. Default = '+str( defaults[ 'preloadModel' ], action = 'store_true' )
    commandLineParser.add_argument( '-e', '--engine', help = 'Must be fairseq, ctranslate2, or transformers.', default = None, type = str )
    commandLineParser.add_argument( '-m', '--modelName', help = 'The name of the model to load. It must be a supported model for that engine.', default = None, type = str )
    commandLineParser.add_argument( '-p', '--modelPath', help = 'For fairseq, the full path to model.pretrain.pt. For CTranslate2, the full path to model.bin.', default = None, type = str )
    commandLineParser.add_argument( '-sl', '--sourceLanguage', help = 'For Sugoi, use a two letter source language code. See: www.loc.gov/standards/iso639-2/php/code_list.php Default = None', default = None, type = str )
    commandLineParser.add_argument( '-tl', '--targetLanguage', help = 'For Sugoi, use a two letter target language code. See: www.loc.gov/standards/iso639-2/php/code_list.php Default = None', default = None, type = str )

    #cache settings
    commandLineParser.add_argument( '-c', '--cache', help = 'Toggle cache setting from default. Enabling cache saves the results of the model for future requests. Default = cache is enabled.', action = 'store_false' )
    commandLineParser.add_argument( '-cp', '--cachePath', help = 'The folder to store the cache. The default is to store cache in the local system appdata or ~/.cache folders.', default = None, type = str )
    commandLineParser.add_argument( '-cfd', '--cacheFileCsvDialect', help = 'The csv dialect used for cache.csv. Default = '+,default = , type = str )

    commandLineParser.add_argument( '-ui', '--uiPath', help = 'The path to the streamlit UI.py. Using streamlit requires installing it via: pip install streamlit', default = None, type = str )
    commandLineParser.add_argument( '-dpm', '--disablePerfMetrics', help = 'Disable tracking and reporting of performance metrics. Default = Enabled.', action = 'store_false' )

    # engine specific CLI values go here
    commandLineParser.add_argument( '-t', '--cpuThreads', help = 'For CTranslate2, the number of CPU threads. Only affects CTranslate2. The default is the number of physical cores if the psutil library is available. The default without psutil is for CTranslate2 to use its internal values. Using psutil requires installing it via: pip install psutil', default = None, type = int )
    commandLineParser.add_argument( '-vm', '--useVMap', help = 'For CTranslate2, enable the use of a vocabulary map. Must be named vmap.txt and exist in the model directory. Default = False.', action = 'store_true' )

    commandLineParser.add_argument( '-ce', '--consoleEncoding', help = 'Specify the encoding used for certain types of stdout. Default = '+defaultConsoleEncoding,default = defaultConsoleEncoding, type = str )
    commandLineParser.add_argument( '-ifeh', '--inputFileErrorHandling', help = 'If the input from files cannot be read perfectly using the specified encoding, what should happen? See: https://docs.python.org/3.8/library/codecs.html#error-handlers Default is to crash the program.', default = defaultInputFileErrorHandling, type = str )
    commandLineParser.add_argument( '-ofeh', '--outputFileErrorHandling', help = 'If the output from files cannot be writen perfectly using the specified encoding, what should happen? See: https://docs.python.org/3.8/library/codecs.html#error-handlers Default is to crash the program.', default = , type = str )
    commandLineParser.add_argument( '-vb', '--verbose', help = 'Print more information.', action = 'store_true' )
    commandLineParser.add_argument( '-d', '--debug', help = 'Print too much information.', action = 'store_true' )
    commandLineParser.add_argument( '-v', '--version', help = 'Print version information and exit.', action = 'store_true' )



    # Parse command line settings.
    commandLineArguments = commandLineParser.parse_args()
    userInput = {}

    userInput[ 'mode' ] = commandLineArguments.mode
    userInput[ 'inputModelFileOrFolder' ] = commandLineArguments.modelPath

    userInput[ 'device' ] = commandLineArguments.device
    userInput[ 'sourceLanguage' ] = commandLineArguments.sourceLanguage
    userInput[ 'targetLanguage' ] = commandLineArguments.targetLanguage
    userInput[ 'sourceSentencePieceModel' ] = commandLineArguments.sourceSentencePieceModel
    userInput[ 'targetSentencePieceModel' ] = commandLineArguments.targetSentencePieceModel

    userInput[ 'preloadModel' ] = commandLineArguments.preloadModel
    userInput[ 'intra_threads' ] = commandLineArguments.cpuThreads
    userInput[ 'use_vmap' ] = commandLineArguments.useVMap
    userInput[ 'perfMetrics' ] = commandLineArguments.disablePerfMetrics

    userInput[ 'cacheEnabled' ] = commandLineArguments.cache
    userInput[ 'uiPath' ] = commandLineArguments.uiPath

    userInput[ 'address' ] = commandLineArguments.address
    userInput[ 'port' ] = commandLineArguments.port

    userInput[ 'cacheFileEncoding' ] = commandLineArguments.cacheFileEncoding
    userInput[ 'consoleEncoding' ] = commandLineArguments.consoleEncoding
    userInput[ 'inputErrorHandling' ] = commandLineArguments.inputFileErrorHandling
    userInput[ 'outputErrorHandling' ] = commandLineArguments.inputFileErrorHandling
    userInput[ 'version' ] = commandLineArguments.version
    userInput[ 'verbose' ] = commandLineArguments.verbose
    userInput[ 'debug' ] = commandLineArguments.debug

    return userInput


def validateInput( defaults=None, userInput=None ):
    #Workaround to fairseq + CPU bug.
    # Update: fairseq seems to hang on any sort of multiprocessing, multithreading, and even simple async + await calls.
    if (mode == 'fairseq') and (device=='cpu') and (preloadModel==False) and (psutilAvailable != True):
        # Then change to preloading the model because there is no way to end the child process reliably otherwise. It hangs after it finishes processing long batches.
        preloadModel = True
        if __name__ == '__main__':
            print( '\n Warning: fairseq + CPU + multiprocessing requires psutil. Install with: \n\n    pip install psutil \n\n Since psutil is not available, preloadModel=True. \n If this behavior is not desired, install psutil.\n')

    settings = userInput.update(defaults) # Is this correct?

    #inputModelFileNameAndPath = None
    #inputModelPathOnly = None
    #inputModelNameWithoutPath = None
    # mode and inputModel will always be used at the CLI as required inputs, so just need to validate they are correct.
    # mode must be fairseq or CTranslate2
    if mode.lower() == 'fairseq':
        try:
            import fairseq
        except ImportError:
            print( 'Error: fairseq was selected for mode but cannot be imported. Please install it with: pip install fairseq' )
            sys.exit( 1 )

        mode = 'fairseq'

        # inputModelFileOrFolder must be a file and it must exist
        verifyThisFileExists( inputModelFileOrFolder , 'inputModelFileOrFolder' )
        #If there is a folder specified, could also try to auto detect a pretrained.pt model for increased flexibility.

        # Create subtypes here using Path library, like path only, extension only. Not sure how they will be used/useful, but can just comment out later.
        inputModelFileNameAndPath = inputModelFileOrFolder
        inputModelPathObject = pathlib.Path( inputModelFileNameAndPath ).absolute()
        inputModelPathOnly = str(inputModelPathObject.parent) # Does not include last /, and this will return one subfolder up if it is called on a folder.
        inputModelNameWithoutPath = inputModelPathObject.name
    elif mode.lower() == 'ctranslate2':
        try:
            import ctranslate2
        except ImportError:
            print( 'Error: ctranslate2 was selected for mode but cannot be imported. Please install it with: pip install ctranslate2' )
            sys.exit( 1 )
        try:
            import sentencepiece
        except ImportError:
            print( 'Error: sentencepiece cannot be imported. Please install sentencepiece with: pip install sentencepiece' )
            sys.exit( 1 )

        mode = 'ctranslate2'
        inputModelFileOrFolderObject = pathlib.Path( inputModelFileOrFolder ).absolute()

        # If the specified path is a file, then get the folder from the str(pathlib.Path(myPath).parent)
        # and then continue to run as normal. ctranslate2 will refuse to load the model if not valid, so do not worry about it.
        if checkIfThisFileExists( inputModelFileOrFolder ) == True:
            inputModelFileNameAndPath=str(inputModelFileOrFolderObject)
            inputModelPathOnly=str(inputModelFileOrFolderObject.parent)
            inputModelNameWithoutPath=inputModelFileOrFolderObject.name
        else:
            #inputModelFileOrFolder must be a folder and it must exist
            # The model must also exist inside of it, but maybe let the ctranslate2 library worry about that? It might have its own code for detecting different ctranslate2 formats or w/e.
            verifyThisFolderExists( inputModelFileOrFolder,'inputModelFileOrFolder' )

            # Create subtypes here using Path library.
            inputModelFileNameAndPath=str(inputModelFileOrFolderObject) + '/' + defaultCTranslate2ModelName
            inputModelPathOnly = str( inputModelFileOrFolderObject )
            # if no model name was specified, then fudge the model name based upon last folder in the path. #Might want to just set this to the defaultCTranslate2ModelName instead.
            inputModelNameWithoutPath = inputModelFileOrFolderObject.parts[ len( inputModelFileOrFolderObject.parts ) - 1 ]
    else:
        print( ( 'Error: mode must be ctranslate2 or fairseq. Mode=' + str( mode ) ).encode( consoleEncoding ) )
        sys.exit( 1 )


    # Now that inputModelNameWithoutPath is known, update some more variables for later use.
    scriptNameWithVersion = currentScriptNameWithoutPath + '/' +__version__
    scriptNameWithVersionDictionary = { 'content' : scriptNameWithVersion }
    modeAndModelName = mode + '/' + inputModelNameWithoutPath
    modeAndModelNameDictionary = { 'content' : modeAndModelName }

    # verify device
    if settings[ 'device' ].lower() == 'cpu':
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
            print( ('Error: Device \'directml\' is only valid for fairseq. Mode=\''+ mode + '\' Current device=\'' + device +'\'').encode(consoleEncoding) )
            sys.exit(1)
        try:
            # https://github.com/microsoft/DirectML/tree/master/PyTorch/1.13
            import torch
            import torch_directml
            dml = torch_directml.device()
        except ImportError:
            print( 'Problem avoided: directml was specified but did not import sucessfully. Consider using anything else, like ctranslate2. Installing directml will trash any existing PyTorch installation. Do not use. Alternatively: pip install torch-directml')
            sys.exit(1)
    else:
        print( ('Error: Unrecognized device=\'' + device + '\' Must be cpu, gpu, cuda, rocm, or directml.' ).encode( consoleEncoding ) )
        sys.exit( 1 )

    if debug == True:
        verbose = True
        import inspect   #Used to print out the name of the current function during execution which is useful when debugging.


















def writeOutCache():
    global translationCacheDictionary

#This turns translationCacheDictionary into a csv file at cacheFilePathAndName.
def clearCache():
    global translationCacheDictionary
    translationCacheDictionary = { }
    print( 'Cleared cache.' )


#if ( __name__ == '__main__' ) and ( cacheEnabled == True ):
def initalizeCache( inputModelFileNameAndPath=None ):
    # Initialize translationCacheDictionary
    translationCacheDictionary = { }
    # Initalize timeCacheWasLastWritten
    timeCacheWasLastWritten = time.perf_counter()
    timeCacheWasLastCleared = time.perf_counter()

    verifyThisFileExists( inputModelFileNameAndPath, 'modelNameAndPath' )
    print( 'Attempting to read cache for model: ' + str( inputModelFileNameAndPath ) )

    # Dump the work of reading the file onto another process so main process does not have to deal with it.

    if modelHashFull == None:
        print( 'Error: Could not generate hash from model file.' + str( inputModelFileNameAndPath )).encode( consoleEncoding ) )
        sys.exit(1)
    modelHash=modelHashFull[:10] # Truncate hash to make the file name more friendly to file system length limitations.

    cacheFilePathOnly=currentScriptPathOnly + '/' + defaultCacheLocation
    cacheFileNameOnly='cache.'+ modelHash + '.csv' #Hardcoded. Maybe add prefix and postfix variables?
    cacheFilePathAndName=cacheFilePathOnly + '/' + cacheFileNameOnly

    if debug == True:#Maybe change this to debug for final settings.
        print( 'modelHash=' + str(modelHash) )
        print( 'cacheFilePathOnly=' + cacheFilePathOnly )
        print( 'cacheFileNameOnly=' + cacheFileNameOnly )
    if verbose == True:
        print( 'cacheFilePathAndName=' + cacheFilePathAndName )

        # Read entries to translationCacheDictionary.
        # If valid then read as normal, but if any error occurs, then print out that there was an error when reading the cache file and just use a new one.
    translationCacheDictionary={}

    if debug == True:
        print( ('translationCacheDictionary=' + str(translationCacheDictionary)).encode(consoleEncoding) )

    print( 'Number of entries loaded into cache: ' + str(len(translationCacheDictionary)) )

    print( (' Cache file not found. Creating a new one at: '+str(cacheFilePathAndName)).encode(consoleEncoding) )



if uiPath != None:
    if checkIfThisFileExists(uiPath) == True:
        uiPath=str( pathlib.Path(uiPath).absolute() )
    else:
        print( 'Warning: Streamlit UI was specified but could not be found:\n')
        print( uiPath.encode(consoleEncoding) )
        print( '' )
        uiPath=None


#Update some internal variables from default values.
bpe=default_bpe
beam_size=default_beam_size
num_hypotheses=default_num_hypotheses
no_repeat_ngram_size=default_no_repeat_ngram_size
#use_vmap=default_use_vmap #Update: Added this to CLI.
inter_threads=default_inter_threads




if ( __name__ == '__main__' ) and (verbose == True) and (mode == 'ctranslate2'):
    print ( 'CTranslate2 CPU threads=' + str(intra_threads) )





# Start app based upon input.
# fairseq will use sourceSentencePieceModel but internally.
if mode == 'fairseq':
    pass
elif mode == 'ctranslate2':
    sourceLanguageProcessor = sentencepiece.SentencePieceProcessor( sourceSentencePieceModel )
    targetLanguageProcessor = sentencepiece.SentencePieceProcessor( targetSentencePieceModel )


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
        print( 'Unspecified error.' )
        sys.exit( 1 )

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

        if ( verbose == True ) and ( perfMetrics==True ):
            processingTime=round( time.perf_counter() - startProcessingTime, 2 )
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
        print( 'Unspecified error.' )
        sys.exit( 1 )


# This function allows run_in_executor() to be added to a taskList, which is a Python list, and then awaiting the taskList.
# That will process all of the entries at once with an instance of concurrent.futures.ProcessPoolExecutor .
# Otherwise, each instance of each task will block the next and also maybe the ioloop depending upon implementation details.
async def proxyTranslateNMT( executor, translateMe ):
    #print( 'pie' * 200 )
    return await asyncio.get_running_loop().run_in_executor( executor, translateNMT, translateMe )


class MainHandler( tornado.web.RequestHandler ):
    async def get( self ):
        print('self.request=' + str( self.request ) )
        if debug == True:
            print( 'Executing: ' + type( self ).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_header( 'Content-Type', 'text/plain' )
        self.set_status( 200 )

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

        # self.args is a dictionary made from self.request.body.
        # self.args[ 'content' ] returns all content specified in the 'content' entry.
        # if that returned item is a list, then self.args[ 'content' ][0] returns the first item in that list.
        if debug == True:
            print( 'self.request.body=' + str( self.request.body ) )

        # Assume input is json and just blindly decode.
        #self.args = tornado.escape.json_decode(self.request.body)
        # Check if input is json, and then code. if content is not application/json, then error out.
        if self.request.headers.get( 'Content-Type') == 'application/json' :
            self.args = tornado.escape.json_decode( self.request.body )
        else:
            print( 'Error: Only json is supported as input currently. Returning.')
            return

        if ( self.args == None ) or ( self.args == '' ):
            print( 'Error: No json contents found in request.body. Returning.')
            return
        if not isinstance( self.args,dict ):
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
            if ( str(self.args[ 'message' ]).lower() == 'close server' ):
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
            rawInput=self.args[ 'content' ]
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
        # The syntax of this is:  tempRequestDictionary.append( ( 'rawEntry', thisValueIsFromCache , translatedData ) )
        #tempRequestDictionary={}
        tempRequestList=[]
        global timeCacheWasLastWritten

        if ( cacheEnabled == True ) and ( len( translationCacheDictionary ) != 0 ):
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
                    # then add entry/i to tempRequestDictionary with thisValueIsFromCache=True
                    #tempRequestDictionary[i]=[True,translationCacheDictionary[i]]
                    tempRequestList.append( [ i, True, translationCacheDictionary[ i ] ] )
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
                    taskList.append( asyncio.create_task( proxyTranslateNMT( executor, translateMe ) ) )

                    # Execute those processes while still in the loop so that executor still exists.
                    #for f in asyncio.as_completed( taskList ):
                    #    finalResults.append( await f )

                    # Alternative smaller and more confusing code to get finalResults. Removing the asterisk * breaks it.
                    # https://stackoverflow.com/questions/36901/what-does-double-star-asterisk-and-star-asterisk-do-for-parameters
                    #finalResults = await asyncio.gather( *taskList )
                    postTranslatedList = await asyncio.gather( *taskList ) # *postTranslatedList supposedly means 'unpack postTranslatedList' which still does not clarify its usage. Why does the assignment break when removing it? Maybe it is not a return object, but the actual stored create_task functions themselves? But in that case, then should not just feeding the raw taskList also work without unpacking? What does asyncio.gather() expect?
                    #https://docs.python.org/3/library/asyncio-task.html#running-tasks-concurrently
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

                #The above returns a list which encapsulates all 1 entries in the taskList. translateNMT itself also returns a list, so there is a [ [ ] ] object returned.
                #Remove the outer list.
                postTranslatedList=postTranslatedList[0]


        if debug == True:
            print( ( 'postTranslatedList=' + str(postTranslatedList) ).encode(consoleEncoding) )
            if cacheEnabled == True:
                print( 'translationCacheDictionary length=' + str(len( translationCacheDictionary )) )

        # Initalize finalOutputList
        finalOutputList = []
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
                        finalOutputList.append( i[2] )
                    # if thisValueIsFromCache == False:
                    elif i[1] == False:
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

# self.set_header( 'Server', 'tornado/' + str( tornado.version ) ) #This is incorrect. Tornado automatically sets this correctly on its own. Example response headers for a 404:
# HTTP/1.1 404 Not Found
# Server: TornadoServer/6.4
# Content-Type: text/html; charset=UTF-8
# Date: Sun, 01 Feb 2020 12:00:00 GMT
# Content-Length: 69


class ReturnVersion( tornado.web.RequestHandler ):
    async def get( self ):
        print( 'self.request=' + str( self.request ) )
        if debug == True:
            print( 'Executing: ' + type( self ).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status( 200 )
        self.set_header( 'Content-Type', 'text/plain' )

        self.write( scriptNameWithVersion )

    async def post( self ):
        print( 'self.request=' + str( self.request ) )
        if debug == True:
            print( 'Executing: ' + type( self ).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status( 200 )
        self.set_header( 'Content-Type', 'application/json' )

        self.write( json.dumps( scriptNameWithVersionDictionary ) )


class ReturnModel( tornado.web.RequestHandler ):
    async def get( self ):
        print( 'self.request=' + str( self.request ) )
        if debug == True:
            print( 'Executing: ' + type(self).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status( 200 )
        self.set_header( 'Content-Type', 'text/plain' )

        self.write( modeAndModelName )

    async def post(self):
        print( 'self.request=' + str( self.request ) )
        if debug == True:
            print( 'Executing: ' + type( self ).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status( 200 )
        self.set_header( 'Content-Type', 'application/json' )

        self.write( json.dumps( modeAndModelNameDictionary ) )


class SaveCache( tornado.web.RequestHandler ):
    async def get( self ):
        print( 'self.request=' + str( self.request ) )
        if debug == True:
            print( 'Executing: ' + type( self ).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status( 200 )
        self.set_header( 'Content-Type', 'text/plain' )

        if cacheEnabled != True:
            self.finish( 'Unable to save cache because cache is not enabled.' )
            return

        global timeCacheWasLastWritten
        #Check timer for cache last written. If timer > 60s, then write out to file.
        if int( time.perf_counter()  - timeCacheWasLastWritten ) > defaultSaveCacheInterval:
            timeCacheWasLastWritten = time.perf_counter()
            try:
                writeOutCache()
                self.finish( 'Cache was written to disk.' )
                return
            except:
                print( 'Warning: An unspecified error occured during writeOutCache()' ) # Print to console.
                self.finish( 'Warning: An unspecified error occured during writeOutCache()' ) # Send error message over HTTP.
                return
        else:
            self.finish( 'Cache was not written to disk. To save cache, please wait up to ' + str( defaultSaveCacheInterval ) + ' seconds.' )
            return

    async def post( self ):
        print( 'self.request=' + str(self.request) )
        if debug == True:
            print( 'Executing: ' + type(self).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status( 200 )
        self.set_header( 'Content-Type', 'application/json' )

        if cacheEnabled != True:
            self.finish( json.dumps( { 'content' : 'Unable to save cache because cache is not enabled.' } ) )
            return

        global timeCacheWasLastWritten
        #Check timer for cache last written. If timer > 60s, then write out to file.
        if int( time.perf_counter()  - timeCacheWasLastWritten ) > defaultSaveCacheInterval:
            timeCacheWasLastWritten = time.perf_counter()
            try:
                writeOutCache()
                self.finish( json.dumps( { 'content' : 'Cache was written to disk.' } ) )
                return
            except:
                print( 'Warning: An unspecified error occured during writeOutCache()' ) # Print to console.
                self.finish( json.dumps( { 'content' : 'Warning: An unspecified error occured during writeOutCache()' } ) ) # Send error message over HTTP.
                return
        else:
            self.finish( json.dumps( { 'content' : 'Cache was not written to disk. To save cache, please wait up to ' + str( defaultSaveCacheInterval ) + ' seconds.' } ) )
            return


class ClearCache( tornado.web.RequestHandler ):
    async def get( self ):
        print( 'self.request=' + str( self.request ) )
        if debug == True:
            print( 'Executing: ' + type( self ).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status( 200 )
        self.set_header( 'Content-Type', 'text/plain' )

        if cacheEnabled != True:
            self.finish( 'Unable to clear cache because cache is not enabled.' )
            return

        global timeCacheWasLastCleared
        #Check timer for cache last written. If timer > 60s, then write out to file.
        if int( time.perf_counter()  - timeCacheWasLastCleared) > defaultMinimumClearCacheInterval:
            timeCacheWasLastCleared = time.perf_counter()
            clearCache()
            self.finish( 'Cache was cleared.' )
            return
        else:
            self.finish( 'Cache was not cleared. To clear cache, please wait up to ' + str( defaultMinimumClearCacheInterval ) + ' seconds.' )
            return

    async def post( self ):
        print( 'self.request=' + str( self.request ) )
        if debug == True:
            print( 'Executing: ' + type(self).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status( 200 )
        self.set_header( 'Content-Type', 'application/json' )

        if cacheEnabled != True:
            self.finish( json.dumps( { 'content': 'Unable to clear cache because cache is not enabled.' } ) )
            return

        global timeCacheWasLastCleared
        #Check timer for cache last written. If timer > 60s, then write out to file.
        if int( time.perf_counter()  - timeCacheWasLastCleared) > defaultMinimumClearCacheInterval:
            timeCacheWasLastCleared = time.perf_counter()
            clearCache()
            self.finish( json.dumps( { 'content' : 'Cache was cleared.' } ) )
            return
        else:
            self.finish( json.dumps( { 'content' : 'Cache was not cleared. To clear cache, please wait up to ' + str( defaultMinimumClearCacheInterval ) + ' seconds.' } ) )
            return


class GetCache( tornado.web.RequestHandler ):
    async def get( self ):
        print( 'self.request=' + str( self.request ) )
        if debug == True:
            print( 'Executing: ' + type( self ).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status( 200 )

        if cacheEnabled != True:
            self.set_header( 'Content-Type', 'application/json' )
            self.finish( json.dumps( { 'content' : 'Unable to send cache because cache is not enabled.' } ) )
            return

        #https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Disposition
        self.set_header( 'Content-Type', 'application/csv' )
        self.set_header( 'Content-Disposition', 'attachment; filename=' + cacheFileNameOnly )

        # This might produce an error if the file has not been written to disk yet.
        # It might be better to read the entire file into memory, as cumbersome as that is, and then send it. That minimizes the potential of writing to the file at the same time as reading it. That wastes a lot of memory that will never be reclaimed by the OS, even if del is explcitly called on the object, however. So, which is better? Which is worse? Oh, the joys of async programming.
        chunkSize = 4194304 #4MB
        with open( cacheFilePathAndName, 'rb' ) as myFileHandle:
            while True:
                chunk = myFileHandle.read( chunkSize )
                if not chunk:
                    break
                try:
                    self.write( chunk )
                    await self.flush()
                    await asyncio.sleep( 0 )
                except:
                    break
                finally:
                    del chunk

    async def post( self ):
        print( 'self.request=' + str( self.request ) )
        if debug == True:
            print( 'Executing: ' + type( self ).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status( 200 )
        self.set_header( 'Content-Type', 'application/json' )

        if cacheEnabled != True:
            self.finish( json.dumps( { 'content': 'Unable to send cache because cache is not enabled.' } ) )
            return

        self.finish( json.dumps( dict( [ ('content' , translationCacheDictionary ) ] ), ensure_ascii=False) )
        return


async def runUI( uiPath ):
    # Might be useful somehow: https://docs.python.org/3.8/library/shlex.html#shlex.quote
    #import subprocess
#    myPath=os.path.join(currentScriptPathOnly,str(pathlib.Path(os.path.join( 'resources', 'webUI.py'))))

    myString='\" --server.address='+address+' -- ' + '--address ' + address + ' --port ' + str(port) + ' --quiet'
    #myString=''

    # This syntax starts a fully independent instance of the UI. Great for stability, but not for managing the subprocess.
    # It works, but stars a new shell Window. Makes it more obvious that it needs to cleanly shut down at least.
#    if platform.system().lower() == 'windows':
#    if sys.platform == 'win32':
#        fullCommand='start "py3translationServer UI - by gdiaz384" streamlit run ' + myPath + myString
#    elif platform.system.lower() == 'linux':
#    elif sys.platform == 'linux':
#    else:
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
    #subprocess.run( fullCommand, capture_output=False, shell=True)
    #subprocess.run( fullCommand )

    # asyncio + exec Does not work.
    #await asyncio.create_subprocess_exec(fullCommand)

    # This works perfectly. So, the solution was to use asyncio with a shell.
    # https://docs.python.org/3/library/asyncio.html
    # https://docs.python.org/3/library/asyncio-subprocess.html
    # https://docs.python.org/3.8/library/asyncio-subprocess.html#asyncio.asyncio.subprocess.Process
    try:
        return await asyncio.create_subprocess_shell( fullCommand )
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
    # The following is an attempt to improve this: https://docs.python.org/3/library/asyncio-runner.html#handling-keyboard-interruption
    # Update ctrl + c handler on Windows. Linux should work mostly as expected without modification.
    # From Shital Shah at https://stackoverflow.com/questions/1364173/stopping-python-using-ctrlc
    # Had to change b=None to no default value, but ctrl+c seems to work more reliably now. Maybe. Still does not work sometimes.
    # The only workaround might be to always launch the .py from its own .cmd and then tell cmd to close.
    # The b in handler also does not always work but setting a default is also error prone.
    # Install with: python -m pip install pywin32
    #def handler( a, b ):
    #    sys.exit( 0 )
    # One alternative is platform.system() which returns 'Windows', 'Darwin', or 'Linux'. Not sure what BSD returns.
    #if platform.system().lower() == 'windows':
    if ( sys.platform == 'win32' ) and ( sys.version_info.minor < 11):
        try:
            # Load different handler function for different Python versions to sometimes improve compatibility for older versions.
            # This maybe sometimes breaks compatibility for newer Python versions, maybe.
            if sys.version_info.minor >= 8:
                def handler( a, b ):
                    sys.exit( 0 )
            else:
                def handler( a, b=None ):
                    sys.exit( 0 )
            import win32api
            win32api.SetConsoleCtrlHandler( handler, True )
        except ImportError:
            pass

    # Set some generic defaults that need to be after the import statments.
    currentScriptPathObject = pathlib.Path( __file__ ).resolve()
    currentScriptNameWithoutPath = currentScriptPathObject.name
    currentScriptNameWithoutPathOrExt = currentScriptPathObject.stem
    usageHelp = ' Usage: ' + currentScriptNameWithoutPath + ' -h'
    #Update path of current script.
    currentScriptPathOnly = str( currentScriptPathObject.parent ) #Does not include last / and this will return one subfolder up if it is called on a folder.

    # Get main program defaults.
    defaults = getDefaults( )
    global consoleEncoding
    consoleEncoding = defaults[ 'consoleEncoding' ]
    localSystemCacheLocationWindows = os.getenv( 'LOCALAPPDATA' ) + '/' + currentScriptNameWithoutPathOrExt + '/cache'
    localSystemCacheLocationLinux = commonFunctions.fixPath( '~/.cache/'+ currentScriptNameWithoutPathOrExt + '/cache' )

    # Create and get command line options based on some hardcoded defaults.
    userInputFromCLI = createAndGetCommandLineOptions( defaults=defaults, usageHelp=usageHelp )
    if userInputFromCLI[ 'version' ] == True:
        print( currentScriptNameWithoutPath + ' ' + __version__ ).encode( consoleEncoding )
        sys.exit( 0 )
    # Get config.ini input, if any.
    userInputFromConfig = readConfig( currentScriptNameWithoutPathOrExt + defaults[ 'configFileExtension' ] )
    if 'program' in userInputFromConfig:
        if 'version' in userInputFromConfig[ 'program' ]:
            if userInputFromConfig[ 'program' ][ 'version' ] == True:
                print( currentScriptNameWithoutPath + ' ' + __version__ ).encode( consoleEncoding )
                sys.exit( 0 )
    userInputFromConfig = commonFunctions.fixTypesInConfig( userInputFromConfig ) # modelDatabase is still a string at this point.
    # Merge CLI input, and config.ini. CLI takes priority.
    programSettings = merge( userInputFromCLI, userInputFromConfig )
    # Validate input.
    programSettings = validateInput( defaults, userInput )
    consoleEncoding = programSettings[ 'consoleEncoding' ]

    if ( perfMetrics == True ) or ( verbose==True ) or ( debug == True ):
        #import time                     # Optional library. Used to calculate performance metrics. #Update, processing time should be optionally reported even if verbose==True, so load it if either of those conditions are true. Debug being true implies that verbose is as well. # Update2. Will need to always import time at some point for cache functionality for delaying writing out cache file for at least 30s, ideally 60s.
        startedLoadingTime = time.perf_counter()


    #defaults[ 'modelDatabase' ] = [ 'fairseq.sugoi', 'ctranslate2.sugoi', 'transformers.marianmt' ]
    # The [ 'modelDatabase' ] list should be turned into a modelDatabase={ } dictionary where each key is a model name mapping to a value. Then, the value should be another dictionary that has these values.
    # [ 'imported' ]=False, [ 'module' ]=importlib.import_module( 'fully.qualified.path.to.module' ), [ 'model' ]=[ 'module' ].Translator(), [ 'available' ]=False, [ 'hash' ]=None,
#>>> modelName=[ 'fairseq.sugoi' ]
#>>> sugoi_module = importlib.import_module( 'resources.engines.'+modelName[ 0 ].split( '.' )[ 0 ]+'.'+ modelName[ 0 ].split( '.' )[ 1 ] )
#>>> modelDatabase[ modelName ][ 'model' ] = sugoi_module.Translator()
    modelDatabase = {}
    # modelName == engine.model as in 'fairseq.sugoi', 'ctranslate2.sugoi', or 'transformers.marianmt'
    for modelName in programSettings[ 'program' ][ 'modelDatabase' ]:
        modelDatabase[ modelName ] = { }
        modelDatabase[ modelName ][ 'name' ] = modelName.split( '.' )[1].strip()
        modelDatabase[ modelName ][ 'engine' ] = modelName.split( '.' )[0].strip()
        modelDatabase[ modelName ][ 'imported' ] = False
        modelDatabase[ modelName ][ 'module' ] = None
        modelDatabase[ modelName ][ 'model' ] = None #model instance
        modelDatabase[ modelName ][ 'available' ] = False # available means model is currently available to use. Should only be true for 1 model at a time.
        modelDatabase[ modelName ][ 'hash' ] = None

        if modelName in programSettings:
            modelDatabase[ modelName ][ 'model' ][ 'settings' ] = settings[ modelName ]
            if not 'modelPath' in modelDatabase[ modelName ][ 'model' ][ 'settings' ]:
                modelDatabase[ modelName ][ 'model' ][ 'settings' ][ 'modelPath' ] = None
            if not 'sourceLanguage' in modelDatabase[ modelName ][ 'model' ][ 'settings' ]:
                modelDatabase[ modelName ][ 'model' ][ 'settings' ][ 'sourceLanguage' ] = None
            if not 'targetLanguage' in modelDatabase[ modelName ][ 'model' ][ 'settings' ]:
                modelDatabase[ modelName ][ 'model' ][ 'settings' ][ 'targetLanguage' ] = None
        else:
            modelDatabase[ modelName ][ 'model' ][ 'settings' ] = { }
            modelDatabase[ modelName ][ 'model' ][ 'settings' ][ 'modelPath' ] = None
            modelDatabase[ modelName ][ 'model' ][ 'settings' ][ 'sourceLanguage' ] = None
            modelDatabase[ modelName ][ 'model' ][ 'settings' ][ 'targetLanguage' ] = None

    for modelName in modelDatabase:
        try:
            modelDatabase[ modelName ][ 'module' ] = importlib.import_module( 'resources.engines.' + modelDatabase[ engine ]+ '.' + modelDatabase[ name ] )
            modelDatabase[ modelName ][ 'imported' ] = True
        except:
            pass

    # Validate logical settings.
    # Make sure either preloadModel == True and it exists, or at least 1 knownModel imported and when it imported modelPath exists. modelPath does not have to point to a model or a file, but it must point to one or the other since the engine determines if a folder is enough.
    # if preloadModel is true, then load cache, also start a new worker process by giving it all of the info it needs to start

    # Print information to inform the user and help with debugging. Print it only in main since otherwise it gets printed out a lot.
    # Always print out mode (fairseq/ctranslate 2)
    print( 'mode=\''+mode + '\'' )
    # Always print out device (cpu, cuda, directml)
    print( 'device=\'' + device + '\'' )
    # Always print out source language and target language
    print( ( 'Source Language=\'' + sourceLanguage + '\'' ).encode(consoleEncoding) )
    print( ( 'Target Language= \''+ targetLanguage + '\'' ).encode(consoleEncoding) )

    if ( verbose == True ) or ( debug == True ):
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


    if cacheEnabled == True:
        initalizeCache()








    if perfMetrics == True:
        print( 'Load time: ' + str( round(time.perf_counter() - startedLoadingTime, 2) ) + ' seconds' )


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

    print( ( currentScriptNameWithoutPath + ' v' + __version__).encode( consoleEncoding ) )
    print( ( currentScriptNameWithoutPath + ' ' + mode + ' ' + device + ' started: http://' + str( address ) + ':' + str( port ) ).encode( consoleEncoding ) )
    # if binding to all addresses, then display the connectable addresses for convenience.
    if ( address == '0.0.0.0' ):
        print( 'http://localhost:' + str(port) )
        import socket
        if sys.platform == 'win32':
            for i in socket.getaddrinfo( socket.gethostname() ,None ):
                #print( 'http://' + str(i[4][0]) + ':' + str(port) )
                temp = str( i[ 4 ][ 0 ] )
                # filter out IPv6 addresses
                if temp.find( ':' ) == -1:
                    print( 'http://' + temp + ':' + str( port ) )
        else: #Linux
            # On windows, this prints an error to stderr and stdout returns an array with a single empty string.
            for i in subprocess.run('hostname -I', shell=True, capture_output='stdout').stdout.decode().strip().split(' '):
                if i.strip() == '':
                    continue
                print( 'http://' + i.strip() + ':' + str( port ) )

    # Update this with: https://www.tornadoweb.org/en/stable/netutil.html Done.
    application.listen( address=address, port=port )

    global uiHandle
    uiHandle=None
    if uiPath != None:
        uiHandle = await runUI(uiPath)
    # uiHandle is an instance of: asyncio.subprocess.Process
    #https://docs.python.org/3.8/library/asyncio-subprocess.html#interacting-with-subprocesses

    await asyncio.Event().wait()


if __name__ == '__main__':
    multiprocessing.freeze_support()
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
            for process in psutil.Process( os.getpid() ).children( recursive=True ):
                #process.send_signal( signal.SIGTERM )
                process.terminate() #Mostly an alias for above code.
            if verbose == True:
                print( 'Info: Child processes found and sent signal.SIGTERM.' )
        except psutil.NoSuchProcess:
            if verbose == True:
                print( 'No child processes.' )
    else:
        # This must be below the psutil code that closes subprocesses or the linking process will not exist for psutil to use. See:
        # https://psutil.readthedocs.io/en/latest/#psutil.Process.children
        if (uiPath != None) and (uiHandle != None):
            # This does not work because it only closes the shell instance, not the streamlit + pythonScript.py subprocess.
            #uiHandle.send_signal(signal.SIGTERM)
            uiHandle.terminate() #This is an alias for the above command but with cross platform support.

    if (mode == 'fairseq') and (device == 'cpu'):
        #print('pie',flush=True)
        #psutilAvailable = False
        if psutilAvailable == True:
            #print('pie2', flush=True)
            psutil.Process( os.getpid() ).send_signal( signal.SIGTERM ) # Suicide. The safer way.
        elif psutilAvailable != True:
            #print('pie3', flush=True)
            os.kill( os.getpid(),signal.SIGTERM ) # Suicide.

    print( 'Program crashed successfully.' )
    sys.exit( 0 )
