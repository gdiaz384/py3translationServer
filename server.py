"""
https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor
https://docs.python.org/3/library/asyncio-eventloop.html#scheduling-delayed-callbacks
https://docs.python.org/3/library/multiprocessing.html#examples
https://docs.python.org/3/library/asyncio-eventloop.html#transferring-files (not valid for Tornado?)
https://docs.python.org/3/library/asyncio-subprocess.html#asyncio.create_subprocess_shell

Optimizations:
- Allow user to input hash for models in the .ini. 'hash_modelName = ' and possibily CLI.
- Allow user to input modelPath for non-default models in the .ini, maybe via 'path_modelName = ' syntax and possibily CLI.
"""
__version__ = '2025.06.24'

# Python Standard Library
import sys                            # sys.exit(), which is mostly for debugging, and search path manipulation for finding libraries reliably.
import os                             # os.getpid(), keeps track of the processID of the subprocess.
import platform                  # Allows checking the platform to implement platform specific code paths for feature parity regardless of platform.
import importlib                  # Import modules using strings, calculated paths, and even if the module's name contains invalid symbols like -.
import pathlib                     # Manipulate paths and check if 
import time                         # Check the amount of time that has passed to keep track of idle cacheWrite and stopProcess times during scheduled callbacks.
import json                         # Read and send json during network i/o.
import glob                         # Allows search for engines when config.ini is not present or incomplete.
import multiprocessing    # multiprocessing.freeze_support() and start a subprocesses that can communicate via multiprocessing.queue's.
import asyncio                   # Create an io loop to allow the main process's main thread to run tasks asyncronously instead of sequentially.
import concurrent.futures # Automatically manage, as in create, feed data, close, spawned asyncronous threads and processes.
import inspect                    # Useful for debugging. 
import random                   # For XSRF prevention cookies.

# 3rd party libraries.
import tornado                  # Async and non-blocking web framework.
import tornado.web
try:
    import psutil                   #Allows for discovering and closing child processes in a platform agnostic way. Also allows setting cputhreads to physical cores instead of logical cores.
    psutilAvailable=True
except ImportError:
    psutilAvailable=False

# Program specific libraries. Used for managing cache and for sharing certain reusable functions among main server and engines.
try:
    import resources.multiLanguageCache as multiLanguageCache
except ImportError:
    sys.path.append( str( pathlib.Path( __file__ ).parent.parent.parent ) )
    import resources.multiLanguageCache as multiLanguageCache
try:
    import resources.commonFunctions as commonFunctions
except ImportError:
    sys.path.append( str( pathlib.Path( __file__ ).parent.parent.parent ) )
    import resources.multiLanguageCache as commonFunctions

consoleEncoding = 'utf-8'
address = 'localhost'
address = '0.0.0.0'
port = 14366
engine = 'ctranslate2'
device = 'cpu'
# The number of seconds that must pass with the model idle before it is unloaded from memory. (60)
unloadTime = 15
defaultProcessesSpawnTechnique = 'spawn'
# The number of seconds that must pass before checking to see if the model process is idle or if cache should be written out.
callbackResolution = 10
relativePathToEnginesFolderFromProgramRoot = 'resources/engines'
# The minimum number of seconds that must pass before cache can be written to disk. (60)
saveCacheInterval = 60
# The minimum number of seconds that must pass before cache can be cleared. (60)
clearCacheInterval = 60
# Keep track of changes to the cache so it is only written out when changes have actually been made to it.
cacheWasUpdated = [ False ]
verbose = True
debug = False
#debug = True
if debug == True:
    verbose = True


# This turns 2 readily available paths into a list where each entry is a dictionary that has engine= and model= keys suitable for use with importlib.import_module().
def getEngines( programRoot, relativePathToEnginesFolderFromProgramRoot ):
    myList = glob.glob( programRoot + '/' + relativePathToEnginesFolderFromProgramRoot +'/**', recursive=True )
    engineList = [ ]
    for candidate in myList:
        candidate = pathlib.Path( candidate )
        if candidate.is_file() == True:
            if candidate.suffix.lower() == '.py':
                engineList.append( str( candidate.resolve() )[ : -3 ] )
    prefix = relativePathToEnginesFolderFromProgramRoot.replace( '/', '.' )
    for counter,engine in enumerate( engineList ):
        if engine.startswith( programRoot ) == False:
            print( ( 'Error parsing engine' + str( engine ) ).encode( consoleEncoding ) )
            continue
        # Cut off the programRoot. The +1 removes the path separator after programRoot.
        engine = engine[ len( programRoot ) + 1 : ]
        if platform.system().lower() == 'windows':
            engine = engine.replace( '\\', '.' )
        else:
            engine = engine.replace( '/', '.' )
        if engine.startswith( prefix ) == False:
            print( ( 'Error parsing engine.' + str( engine ) ).encode( consoleEncoding ) )
            continue
        # Cut off prefix. The +1 removes the path separator after prefix.
        engine = engine[ len( prefix ) + 1 : ]
        # Transform into list based upon first remaining . in the path.
        engine = engine.split( '.', maxsplit=1 )
        # { 'engine' : 'ctranslate2', 'modelGroup' : 'sugoi' }
        engineList[ counter ] = { 'engine' : engine[ 0 ], 'modelGroup' : engine[ 1 ] }
    return engineList



#allModelsRaw is a mapping of modulePath: module.models. trimAllModelsRaw removes modelPath and the tokenizer settings, returning only generic non-sensitive values about all possible models that can be loaded.
#allModelsTrimmed = trimAllModelsRaw( allModelsRaw )
#for key,item in allModelsTrimmed.entries():
#    print( key, len( allModelsTrimmed[ key ] ) )
def trimAllModelsRaw( allModelsRaw ):
    trimmed = { }
    for modulePath in allModelsRaw:
        if allModelsRaw[ modulePath ] == None:
            continue
        if allModelsRaw[ modulePath ][ 'models' ] == None:
            continue
        #if modulePath == 'devices':
        #    continue
        if len( allModelsRaw[ modulePath ][ 'models' ] ) == 0:
            continue
        trimmed[ modulePath ] = { }
        trimmed[ modulePath ][ 'models' ] = { }
        engine, modelGroup = modulePath.split( '.', maxsplit=1 )
        for modelName in allModelsRaw[ modulePath ][ 'models' ]:
            if modelName == 'defaultModel':
                continue
            trimmed[ modulePath ][ 'models' ][ modelName ] = { }
            trimmed[ modulePath ][ 'models' ][ modelName ][ 'engine' ] = engine
            trimmed[ modulePath ][ 'models' ][ modelName ][ 'modelGroup' ] = modelGroup
            trimmed[ modulePath ][ 'models' ][ modelName ][ 'sourceLanguages' ] = allModelsRaw[ modulePath ][ 'models' ][ modelName ][ 'sourceLanguages' ]
            trimmed[ modulePath ][ 'models' ][ modelName ][ 'targetLanguages' ] = allModelsRaw[ modulePath ][ 'models' ][ modelName ][ 'targetLanguages' ]
            trimmed[ modulePath ][ 'models' ][ modelName ][ 'sourceLanguage' ] = allModelsRaw[ modulePath ][ 'models' ][ modelName ][ 'sourceLanguage' ]
            trimmed[ modulePath ][ 'models' ][ modelName ][ 'targetLanguage' ] = allModelsRaw[ modulePath ][ 'models' ][ modelName ][ 'targetLanguage' ]
            trimmed[ modulePath ][ 'models' ][ modelName ][ 'modelURLs' ] = allModelsRaw[ modulePath ][ 'models' ][ modelName ][ 'modelURLs' ]
            trimmed[ modulePath ][ 'models' ][ modelName ][ 'description' ] = allModelsRaw[ modulePath ][ 'models' ][ modelName ][ 'description' ]
        if ( 'devices' in allModelsRaw[ modulePath ] ) == True:
            trimmed[ modulePath ][ 'devices' ] = allModelsRaw[ modulePath ][ 'devices' ]
        else:
            trimmed[ modulePath ][ 'devices' ] = None
    return trimmed


# validateModels makes sure every model exists, fixes paths by making sure any modelPath that points to a folder get fixed to point to .bin files if possible, and returns a summary of modelDatabase[ modulePath ][ 'models' ] where each model entry has been validated and lacks sensitive or internal information like modelPath and tokenizer settings.
#modelDatabaseValidated = { }
#modelDatabaseValidated[ 'defaultModelGroup' ] = None
#modelDatabaseValidated[ modulePath] = { }
#modelDatabaseValidated[ modulePath ][ 'settings' ][ 'modelName' ] =       # The default model for the group.
#modelDatabaseValidated[ modulePath][ 'settings' ][ sourceLanguage ]   # The default sourceLanguage for the group.
#modelDatabaseValidated[ modulePath][ 'settings' ][ targetLanguage ]    # The default targetLanguage for the group.
#modelDatabaseValidated[ modulePath ][ 'models' ]= { }           # Each entry is a valid model and has these keys: sourceLanguages, targetLanguages, modelURLs, description.
#modelDatabaseValidated = validateModels( modelDatabase )
def validateModels( modelDatabase ):
    modelDatabaseValidated = { }
    #modelDatabase[ 'ctranslate2.sugoi' ][ 'settings' ][ 'modelName' ] # This has the default settings for the group, the modelName of the model
    #modelDatabase[ 'ctranslate2.sugoi' ][ 'settings' ][ 'modelPath' ] # This has the default settings for the group, the modelPath to the default model.
    #modelDatabase[ modulePath ][ 'models' ][ model ][ 'modelPath' ] #This is the path for a particular model.

    # Cycle through every group in the modelDatabase.
    #counter = 0
    for modulePath in modelDatabase:
        if ( modulePath == 'defaultModelGroup' ) or ( modulePath == 'active' ):
            continue
        #print( modulePath, modelDatabase[ modulePath ][ 'imported' ] )
        if modelDatabase[ modulePath ][ 'imported' ] == False:
            continue

        #if counter == 0:
        #    counter += 1
        # Before cycling through the models found in the engine, modelDatabase[ modulePath ][ 'models' ], check if the defaults from the.ini are valid, modelDatabase[ modulePath ][ 'settings' ].
        if 'modelName' in modelDatabase[ modulePath ][ 'settings' ]:
            if isinstance( modelDatabase[ modulePath ][ 'settings' ][ 'modelName' ], str ) == True:
                if ( modelDatabase[ modulePath ][ 'settings' ][ 'modelName' ] in modelDatabase[ modulePath ][ 'models' ] ) == True:
                    if ( 'modelPath' in modelDatabase[ modulePath ][ 'settings' ] ) == True:
                        #print( modelDatabase[ modulePath ][ 'settings' ]['modelPath'] )
                        if isinstance( modelDatabase[ modulePath ][ 'settings' ][ 'modelPath' ], str ) == True:
                            # This accepts a huggingface model identifier, including as a url, uses huggingface_hub.try_to_load_from_cache() to turn it into a c:\filesystem\path\to\model.
                            modelDatabase[ modulePath ][ 'settings' ][ 'modelPath' ] = commonFunctions.resolveHuggingfaceUrlToPath( modelDatabase[ modulePath ][ 'settings' ][ 'modelPath' ] )
                            modelDatabase[ modulePath ][ 'settings' ][ 'modelPath' ] = commonFunctions.fixPath( modelDatabase[ modulePath ][ 'settings' ][ 'modelPath' ], basePath=str( pathlib.Path( __file__ ).parent ) )
                            if pathlib.Path( modelDatabase[ modulePath ][ 'settings' ][ 'modelPath' ] ).exists() == True:
                                if pathlib.Path( modelDatabase[ modulePath ][ 'settings' ][ 'modelPath' ] ).is_dir() == True:
                                    for filename in modelDatabase[ modulePath ][ 'module' ].defaultModelBinNames:
                                        fixedPath = modelDatabase[ modulePath ][ 'settings' ][ 'modelPath' ] + '/' + filename
                                        if pathlib.Path( fixedPath ).is_file() == True:
                                            modelDatabase[ modulePath ][ 'settings' ][ 'modelPath' ] = fixedPath
                                            break
                                #print( modelDatabase[ modulePath ][ 'settings' ][ 'modelPath' ] )
                                if pathlib.Path( modelDatabase[ modulePath ][ 'settings' ][ 'modelPath' ] ).is_file() == True:
                                    # Add the modulePath to modelDatabaseValidated.
                                    modelDatabaseValidated[ modulePath ] = { }
                                    modelDatabaseValidated[ modulePath ][ 'settings' ] = { }
                                    modelDatabaseValidated[ modulePath ][ 'settings' ][ 'modelName' ] =  modelDatabase[ modulePath ][ 'settings' ][ 'modelName' ]

                                    modelDatabaseValidated[ modulePath ][ 'models' ] =  { }
                                    # Add the model to modelDatabaseValidated.
                                    model = modelDatabase[ modulePath ][ 'settings' ][ 'modelName' ]
                                    modelDatabaseValidated[ modulePath ][ 'models' ][ model ] = { }
                                    modelDatabaseValidated[ modulePath ][ 'models' ][ model ][ 'sourceLanguages' ] =  modelDatabase[ modulePath ][ 'models' ][ model ][ 'sourceLanguages' ]
                                    modelDatabaseValidated[ modulePath ][ 'models' ][ model ][ 'targetLanguages' ] =  modelDatabase[ modulePath ][ 'models' ][ model ][ 'targetLanguages' ]

                                    # Module specific overrides take precedence over module wide defaults, even if the user specified a module default in the .ini or cli.
                                    # First, take the sourceLanguage from the user specified settings for that exact model in the .ini, if appropriate. Always validate against [ 'sourceLanguages' ].
                                    # Then take the sourceLanguage from the model's defaults obtained from loading [ 'module' ].models, if appropriate as modelDatabase[ modulePath ][ 'models' ][ model ][ 'sourceLanguage' ].
                                    # Then take the sourceLanguage from the module's default language specified in the .ini, modelDatabase[ modulePath ][ 'settings' ][ 'sourceLanguage' ]
                                    # Then take the sourceLanguage from the module's defaults, if appropriate.
                                    # Last resort is to just blindly set it to the first option from sourceLanguages.
                                    sourceLanguage = None
                                    if ( 'iniSettings' in modelDatabase[ modulePath ][ 'models' ][ model ] ) == True:
                                        if ( 'sourceLanguage' in modelDatabase[ modulePath ][ 'models' ][ model ][ 'iniSettings' ] ) == True:
                                            if ( modelDatabase[ modulePath ][ 'models' ][ model ][ 'iniSettings' ][ 'sourceLanguage' ] in modelDatabase[ modulePath ][ 'models' ][ model ][ 'sourceLanguages' ] ) == True:
                                                sourceLanguage = modelDatabase[ modulePath ][ 'models' ][ model ][ 'iniSettings' ][ 'sourceLanguage' ]
                                    if sourceLanguage == None:
                                        if ( 'sourceLanguage' in modelDatabase[ modulePath ][ 'models' ][ model ] ) == True:
                                            if ( modelDatabase[ modulePath ][ 'models' ][ model ][ 'sourceLanguage' ] in modelDatabase[ modulePath ][ 'models' ][ model ][ 'sourceLanguages' ] ) == True:
                                                sourceLanguage = modelDatabase[ modulePath ][ 'models' ][ model ][ 'sourceLanguage' ]
                                    if sourceLanguage == None:
                                        if ( 'sourceLanguage' in modelDatabase[ modulePath ][ 'settings' ] ) == True:
                                            if ( modelDatabase[ modulePath ][ 'settings' ][ 'sourceLanguage' ] in modelDatabase[ modulePath ][ 'models' ][ model ][ 'sourceLanguages' ] ) == True:
                                                sourceLanguage = modelDatabase[ modulePath ][ 'settings' ][ 'sourceLanguage' ]
                                    if sourceLanguage == None:
                                        if ( modelDatabase[ modulePath ][ 'module' ].defaultSourceLanguage in modelDatabase[ modulePath ][ 'models' ][ model ][ 'sourceLanguages' ] ) == True:
                                            sourceLanguage = modelDatabase[ modulePath ][ 'module' ].defaultSourceLanguage
                                    if sourceLanguage == None:
                                        sourceLanguage = modelDatabase[ modulePath ][ 'models' ][ model ][ 'sourceLanguages' ][ 0 ]
                                    assert( sourceLanguage != None )
                                    modelDatabaseValidated[ modulePath ][ 'models' ][ model ][ 'sourceLanguage' ] = sourceLanguage
                                    modelDatabase[ modulePath ][ 'models' ][ model ][ 'sourceLanguage' ] = sourceLanguage

                                    targetLanguage = None
                                    if ( 'iniSettings' in modelDatabase[ modulePath ][ 'models' ][ model ] ) == True:
                                        if ( 'targetLanguage' in modelDatabase[ modulePath ][ 'models' ][ model ][ 'iniSettings' ] ) == True:
                                            if ( modelDatabase[ modulePath ][ 'models' ][ model ][ 'iniSettings' ][ 'targetLanguage' ] in modelDatabase[ modulePath ][ 'models' ][ model ][ 'targetLanguages' ] ) == True:
                                                targetLanguage = modelDatabase[ modulePath ][ 'models' ][ model ][ 'iniSettings' ][ 'targetLanguage' ]
                                    if targetLanguage == None:
                                        if ( 'targetLanguage' in modelDatabase[ modulePath ][ 'models' ][ model ] ) == True:
                                            if ( modelDatabase[ modulePath ][ 'models' ][ model ][ 'targetLanguage' ] in modelDatabase[ modulePath ][ 'models' ][ model ][ 'targetLanguages' ] ) == True:
                                                targetLanguage = modelDatabase[ modulePath ][ 'models' ][ model ][ 'targetLanguage' ]
                                    if targetLanguage == None:
                                        if ( 'targetLanguage' in modelDatabase[ modulePath ][ 'settings' ] ) == True:
                                            if ( modelDatabase[ modulePath ][ 'settings' ][ 'targetLanguage' ] in modelDatabase[ modulePath ][ 'models' ][ model ][ 'targetLanguages' ] ) == True:
                                                targetLanguage = modelDatabase[ modulePath ][ 'settings' ][ 'targetLanguage' ]
                                    if targetLanguage == None:
                                        if ( modelDatabase[ modulePath ][ 'module' ].defaultSourceLanguage in modelDatabase[ modulePath ][ 'models' ][ model ][ 'targetLanguages' ] ) == True:
                                            targetLanguage = modelDatabase[ modulePath ][ 'module' ].defaultTargetLanguage
                                    if targetLanguage == None:
                                        targetLanguage = modelDatabase[ modulePath ][ 'models' ][ model ][ 'targetLanguages' ][ 0 ]
                                    assert( targetLanguage != None )
                                    modelDatabaseValidated[ modulePath ][ 'models' ][ model ][ 'targetLanguage' ] = targetLanguage
                                    modelDatabase[ modulePath ][ 'models' ][ model ][ 'targetLanguage' ] = targetLanguage

                                    # Write back changes to the default model.
                                    modelDatabaseValidated[ modulePath ][ 'settings' ][ 'sourceLanguage' ] = sourceLanguage
                                    modelDatabaseValidated[ modulePath ][ 'settings' ][ 'targetLanguage' ] = targetLanguage
                                    modelDatabase[ modulePath ][ 'settings' ][ 'sourceLanguage' ] = sourceLanguage
                                    modelDatabase[ modulePath ][ 'settings' ][ 'targetLanguage' ] = targetLanguage

                                    modelDatabaseValidated[ modulePath ][ 'models' ][ model ][ 'modelURLs' ] =  modelDatabase[ modulePath ][ 'models' ][ model ][ 'modelURLs' ]
                                    modelDatabaseValidated[ modulePath ][ 'models' ][ model ][ 'description' ] =  modelDatabase[ modulePath ][ 'models' ][ model ][ 'description' ]

        for model in modelDatabase[ modulePath ][ 'models' ]:
            if model == 'defaultModel':
                continue
            #print( modelDatabase[ modulePath ][ 'engine' ], model )
            modelDatabase[ modulePath ][ 'models' ][ model ][ 'modelPath' ] = commonFunctions.resolveHuggingfaceUrlToPath( modelDatabase[ modulePath ][ 'models' ][ model ][ 'modelPath' ] )
            modelDatabase[ modulePath ][ 'models' ][ model ][ 'modelPath' ] = commonFunctions.fixPath( modelDatabase[ modulePath ][ 'models' ][ model ][ 'modelPath' ], basePath=str( pathlib.Path( __file__ ).parent ) )
            #print( modelDatabase[ modulePath ][ 'models' ][ model ][ 'modelPath' ] )
            if modelDatabase[ modulePath ][ 'models' ][ model ][ 'modelPath' ] == None:
                continue
            if pathlib.Path( modelDatabase[ modulePath ][ 'models' ][ model ][ 'modelPath' ] ).exists() == False:
                continue
            if pathlib.Path( modelDatabase[ modulePath ][ 'models' ][ model ][ 'modelPath' ] ).is_dir() == True:
                for filename in modelDatabase[ modulePath ][ 'module' ].defaultModelBinNames:
                    fixedPath = modelDatabase[ modulePath ][ 'models' ][ model ][ 'modelPath' ] + '/' + filename
                    if pathlib.Path( fixedPath ).is_file() == True:
                        modelDatabase[ modulePath ][ 'models' ][ model ][ 'modelPath' ] = fixedPath
                        break
            if pathlib.Path( modelDatabase[ modulePath ][ 'models' ][ model ][ 'modelPath' ] ).is_file() == False:
                continue
            # So the modelPath exists, and it is a file. Initialize the modulePath first.
            if ( modulePath in modelDatabaseValidated ) == False:
                modelDatabaseValidated[ modulePath ] = { }
                modelDatabaseValidated[ modulePath ][ 'settings' ] =  { }
                modelDatabaseValidated[ modulePath ][ 'settings' ][ 'modelName' ] =  modelDatabase[ modulePath ][ 'settings' ][ 'modelName' ]
                modelDatabaseValidated[ modulePath ][ 'settings' ][ 'sourceLanguage' ] =  modelDatabase[ modulePath ][ 'settings' ][ 'sourceLanguage' ]
                modelDatabaseValidated[ modulePath ][ 'settings' ][ 'targetLanguage' ] =  modelDatabase[ modulePath ][ 'settings' ][ 'targetLanguage' ]
                modelDatabaseValidated[ modulePath ][ 'models' ] =  { }
            # Add the model to modelDatabaseValidated.
            if ( model in modelDatabaseValidated[ modulePath ][ 'models' ] ) == False:
                modelDatabaseValidated[ modulePath ][ 'models' ][ model ] = { }
                modelDatabaseValidated[ modulePath ][ 'models' ][ model ][ 'sourceLanguages' ] =  modelDatabase[ modulePath ][ 'models' ][ model ][ 'sourceLanguages' ]
                modelDatabaseValidated[ modulePath ][ 'models' ][ model ][ 'targetLanguages' ] =  modelDatabase[ modulePath ][ 'models' ][ model ][ 'targetLanguages' ]

                # First, take the sourceLanguage from the user specified settings in the .ini, if appropriate. Always validate against [ 'sourceLanguages' ].
                # Then take the sourceLanguage from the model's defaults obtained from loading [ 'module' ].models, if appropriate as modelDatabase[ modulePath ][ 'models' ][ model ][ 'sourceLanguage' ].
                # Then take the sourceLanguage from the module's defaults, if appropriate.
                # Last resort is to just blindly set it to the first option from sourceLanguages.
                sourceLanguage = None
                if ( 'iniSettings' in modelDatabase[ modulePath ][ 'models' ][ model ] ) == True:
                    if ( 'sourceLanguage' in modelDatabase[ modulePath ][ 'models' ][ model ][ 'iniSettings' ] ) == True:
                        if ( modelDatabase[ modulePath ][ 'models' ][ model ][ 'iniSettings' ][ 'sourceLanguage' ] in modelDatabase[ modulePath ][ 'models' ][ model ][ 'sourceLanguages' ] ) == True:
                            sourceLanguage = modelDatabase[ modulePath ][ 'models' ][ model ][ 'iniSettings' ][ 'sourceLanguage' ]
                if sourceLanguage == None:
                    if ( 'sourceLanguage' in modelDatabase[ modulePath ][ 'models' ][ model ] ) == True:
                        if ( modelDatabase[ modulePath ][ 'models' ][ model ][ 'sourceLanguage' ] in modelDatabase[ modulePath ][ 'models' ][ model ][ 'sourceLanguages' ] ) == True:
                            sourceLanguage = modelDatabase[ modulePath ][ 'models' ][ model ][ 'sourceLanguage' ]
                if sourceLanguage == None:
                    if ( 'sourceLanguage' in modelDatabase[ modulePath ][ 'settings' ] ) == True:
                        if ( modelDatabase[ modulePath ][ 'settings' ][ 'sourceLanguage' ] in modelDatabase[ modulePath ][ 'models' ][ model ][ 'sourceLanguages' ] ) == True:
                            sourceLanguage = modelDatabase[ modulePath ][ 'settings' ][ 'sourceLanguage' ]
                if sourceLanguage == None:
                    if ( modelDatabase[ modulePath ][ 'module' ].defaultSourceLanguage in modelDatabase[ modulePath ][ 'models' ][ model ][ 'sourceLanguages' ] ) == True:
                        sourceLanguage = modelDatabase[ modulePath ][ 'module' ].defaultSourceLanguage
                if sourceLanguage == None:
                    sourceLanguage = modelDatabase[ modulePath ][ 'models' ][ model ][ 'sourceLanguages' ][ 0 ]
                assert( sourceLanguage != None )
                modelDatabaseValidated[ modulePath ][ 'models' ][ model ][ 'sourceLanguage' ] = sourceLanguage
                modelDatabase[ modulePath ][ 'models' ][ model ][ 'sourceLanguage' ] = sourceLanguage

                targetLanguage = None
                if ( 'iniSettings' in modelDatabase[ modulePath ][ 'models' ][ model ] ) == True:
                    if ( 'targetLanguage' in modelDatabase[ modulePath ][ 'models' ][ model ][ 'iniSettings' ] ) == True:
                        if ( modelDatabase[ modulePath ][ 'models' ][ model ][ 'iniSettings' ][ 'targetLanguage' ] in modelDatabase[ modulePath ][ 'models' ][ model ][ 'targetLanguages' ] ) == True:
                            targetLanguage = modelDatabase[ modulePath ][ 'models' ][ model ][ 'iniSettings' ][ 'targetLanguage' ]
                if targetLanguage == None:
                    if ( 'targetLanguage' in modelDatabase[ modulePath ][ 'models' ][ model ] ) == True:
                        if ( modelDatabase[ modulePath ][ 'models' ][ model ][ 'targetLanguage' ] in modelDatabase[ modulePath ][ 'models' ][ model ][ 'targetLanguages' ] ) == True:
                            targetLanguage = modelDatabase[ modulePath ][ 'models' ][ model ][ 'targetLanguage' ]
                if targetLanguage == None:
                    if ( 'targetLanguage' in modelDatabase[ modulePath ][ 'settings' ] ) == True:
                        if ( modelDatabase[ modulePath ][ 'settings' ][ 'targetLanguage' ] in modelDatabase[ modulePath ][ 'models' ][ model ][ 'targetLanguages' ] ) == True:
                            targetLanguage = modelDatabase[ modulePath ][ 'settings' ][ 'targetLanguage' ]
                if targetLanguage == None:
                    if ( modelDatabase[ modulePath ][ 'module' ].defaultSourceLanguage in modelDatabase[ modulePath ][ 'models' ][ model ][ 'targetLanguages' ] ) == True:
                        targetLanguage = modelDatabase[ modulePath ][ 'module' ].defaultTargetLanguage
                if targetLanguage == None:
                    targetLanguage = modelDatabase[ modulePath ][ 'models' ][ model ][ 'targetLanguages' ][ 0 ]
                assert( targetLanguage != None )
                modelDatabaseValidated[ modulePath ][ 'models' ][ model ][ 'targetLanguage' ] = targetLanguage
                modelDatabase[ modulePath ][ 'models' ][ model ][ 'targetLanguage' ] = targetLanguage

                modelDatabaseValidated[ modulePath ][ 'models' ][ model ][ 'modelURLs' ] =  modelDatabase[ modulePath ][ 'models' ][ model ][ 'modelURLs' ]
                modelDatabaseValidated[ modulePath ][ 'models' ][ model ][ 'description' ] =  modelDatabase[ modulePath ][ 'models' ][ model ][ 'description' ]

        # Reset the counter at the end of processing a model group so the next model group can get processed.
        #counter = 0
        if ( modulePath in modelDatabaseValidated ) == False:
            continue

        #assert( modelDatabase[ modulePath ][ modelName ] )# sourceLanguage, targetLanguage != None
        # Now that every model has finished processing, make sure the default model for the modulePath exists, the default sourceLanguage and default targetLanguage are correct, and update the devices.
        # Make sure default model for modulePath exists.
        updateModelName = False
        if modelDatabaseValidated[ modulePath ][ 'settings' ][ 'modelName' ] == None:
            updateModelName = True
        # if the modelName set for a modelDatabase[ modulePath ] is not a valid entry, then update it
        elif ( modelDatabaseValidated[ modulePath ][ 'settings' ][ 'modelName' ] in modelDatabaseValidated[ modulePath ][ 'models' ] ) == False:
            updateModelName = True
        if updateModelName == True:
            for model in modelDatabaseValidated[ modulePath ][ 'models' ]:
                # Blindly set the default to the first group.
                modelDatabaseValidated[ modulePath ][ 'settings' ][ 'modelName' ] = model
                modelDatabase[ modulePath ][ 'settings' ][ 'modelName' ] = model
                modelDatabase[ modulePath ][ 'settings' ][ 'modelPath' ] = modelDatabase[ modulePath ][ 'models' ][ model ][ 'modelPath' ]
                assert( pathlib.Path( modelDatabase[ modulePath ][ 'settings' ][ 'modelPath' ] ).is_file() == True )
                break
        # Update the modulePath's default sourceLanguage and targetLanguage based upon the default modelName. If the user does not want this to happen, then they must set valid defaults, as in a valid modelName, modelPath, targetLanguage, and targetLanguage which is a reasonable requirement.
        # Write back changes to main modelDatabase as well.
        defaultModelForGroup=modelDatabaseValidated[ modulePath ][ 'settings' ][ 'modelName' ]
        if ( modelDatabaseValidated[ modulePath ][ 'settings' ][ 'sourceLanguage' ] == None ) or ( ( modelDatabaseValidated[ modulePath ][ 'settings' ][ 'sourceLanguage' ] in modelDatabaseValidated[ modulePath ][ 'models' ][ defaultModelForGroup ][ 'sourceLanguages' ] ) == False ):
            modelDatabaseValidated[ modulePath ][ 'settings' ][ 'sourceLanguage' ] = modelDatabaseValidated[ modulePath ][ 'models' ][ defaultModelForGroup ][ 'sourceLanguages' ][ 0 ]
            modelDatabase[ modulePath ][ 'settings' ][ 'sourceLanguage' ] = modelDatabaseValidated[ modulePath ][ 'models' ][ defaultModelForGroup ][ 'sourceLanguages' ][ 0 ]
        if ( modelDatabaseValidated[ modulePath ][ 'settings' ][ 'targetLanguage' ] == None ) or ( ( modelDatabaseValidated[ modulePath ][ 'settings' ][ 'targetLanguage' ] in modelDatabaseValidated[ modulePath ][ 'models' ][ defaultModelForGroup ][ 'targetLanguages' ] ) == False ):
            modelDatabaseValidated[ modulePath ][ 'settings' ][ 'targetLanguage' ] = modelDatabaseValidated[ modulePath ][ 'models' ][ defaultModelForGroup ][ 'targetLanguages' ][ 0 ]
            modelDatabase[ modulePath ][ 'settings' ][ 'targetLanguage' ] = modelDatabaseValidated[ modulePath ][ 'models' ][ defaultModelForGroup ][ 'targetLanguages' ][ 0 ]
        modelDatabaseValidated[ modulePath ][ 'settings' ][ 'devices' ] = modelDatabase[ modulePath ][ 'settings' ][ 'devices' ]

    # Now that every modulePath is done processing, make sure the defaults, defaultModelGroup, for both modelDatabase and for modelDatabaseValidated are valid and match.
    if ( modelDatabase[ 'defaultModelGroup' ] in modelDatabaseValidated ) == True:
        modelDatabaseValidated[ 'defaultModelGroup' ] = modelDatabase[ 'defaultModelGroup' ]
    #elif ( modelDatabase[ 'defaultModelGroup' ] == None ) or ( ( modelDatabase[ 'defaultModelGroup' ] in modelDatabaseValidated ) == False ):
    else:
        for modulePath in modelDatabaseValidated:
            modelDatabaseValidated[ 'defaultModelGroup' ] = modulePath
            modelDatabase[ 'defaultModelGroup' ] = modulePath
            break

    #print(modelDatabase)
    return modelDatabaseValidated


# Cache hits can either be added one by one as they are needed or in a batch that blindly adds everything after translating the results. If done one by one, then this function is pointless. If done as a batch, this function needs to be reimplemented to support that. If done as a batch, the target function needs to decide whether or not to update the cache based upon the newest generated data blindly, conditionally if target == None, or to discard updates. Currently, multiLanguageCache().add() is performing the updates blindly. This behavior in mlc().add maybe should be adjusted.
async def addToCache( rawString, translatedString, cache, sourceLanguageCode, targetLanguageCode ):
    #def add( self, untranslatedString, translatedString, sourceLanguageCode=None, targetLanguageCode=None ):
    cache.add( untranslatedString=rawString, translatedString=translatedString, sourceLanguageCode=sourceLanguageCode, targetLanguageCode=targetLanguageCode )


# This function checks cache for entries in [ rawText ]. It outputs 2 lists,
# translateMe, the list of items to translate,
# tempRequestList is a master list containing the [ untranslatedString, a boolean representing if it was found in cache or not, translatedString ]
# In tempRequestList, translatedString is only present if the boolean == True.
async def getFromCache( rawText, cache, sourceLanguageCode, targetLanguageCode ):
    translateMe = [ ]
    tempRequestList = [ ]
    # This uses the cache object's internal logic for processing cache as a minor performance optimization. Not clear if worth it. Unlikely.
    languagePair = sourceLanguageCode + '.' + targetLanguageCode
    if ( languagePair in cache.cache ) == False:
        for i in rawText:
            tempRequestList.append( ( i, False ) )
        translateMe = rawText
    else:
        for i in rawText:
            if i in cache.cache[ languagePair ]:
                tempRequestList.append( ( i, True, cache.cache[ languagePair ][ i ] ) )
            else:
                tempRequestList.append( ( i, False ) )
                translateMe.append( i )

    return translateMe, tempRequestList


# This loads the model and translates the text.
# https://docs.python.org/3/library/multiprocessing.html#examples
def translateNMT( inputQueue, outputQueue ):
    # job = ( translateMe data, modelSettings={ engine, modelGroup, modelName, modelPath, device, sourceLanguage, targetLanguage, hash, tokenizer settings } )
    for counter,job in enumerate( iter( inputQueue.get, 'stopProcess' ) ):
        # Debug code.
        #print( job )

        # Unpack job.
        translateMe = job[ 0 ]
        modelSettings = job[ 1 ]

        # Debug code.
        #print( modelSettings )

        # Initialize translator. This loads the model at modelPath into memory using modelSettings.
        if counter == 0:
            if modelSettings[ 'device' ] != None:
                print( 'Loading \'' + modelSettings[ 'engine' ] + '.' + modelSettings[ 'modelGroup' ] + '/' + modelSettings[ 'modelName' ] +'\' on device ' + str( modelSettings[ 'device' ] ) )
            else:
                print( 'Loading \'' + modelSettings[ 'engine' ] + '.' + modelSettings[ 'modelGroup' ] + '/' + modelSettings[ 'modelName' ] + '\' on device autodetect' )
            try:
                importPath = relativePathToEnginesFolderFromProgramRoot.replace( '/', '.' ) + '.' + modelSettings[ 'engine' ] + '.' + modelSettings[ 'modelGroup' ]
                #print( ( 'Importing: ' + importPath ).encode( consoleEncoding ) )
                module = importlib.import_module( importPath )
                translator = module.Translator( modelName=modelSettings[ 'modelName' ], modelPath=modelSettings[ 'modelPath' ], device=modelSettings[ 'device' ], modelSettings=modelSettings )
            except:
                print( 'Error initializing model based on the following input:' )
                print( [ importPath, modelSettings ] )
                response = { }
                response[ 'translatedData' ] = None
                response[ 'engine' ] = modelSettings[ 'engine' ]
                response[ 'modelGroup' ] = modelSettings[ 'modelGroup' ]
                response[ 'modelName' ] = modelSettings[ 'modelName' ]
                response[ 'device' ] = modelSettings[ 'device' ]
                if hash in modelSettings:
                    response[ 'hash' ] = modelSettings[ 'hash' ]
                else:
                    response[ 'hash' ] = None
                response[ 'processID' ] = os.getpid()
                translator = None
                outputQueue.put( response )
                raise
                return

        # Translate the list of strings.
        if translateMe[ 0 ] == 'dummyRequestForPreloadModel':
            result = [ 'dummyRequestForPreloadModel' ]
        else:
            try:
                if debug == True:
                    print( 'Translating data using the following settings:' )
                    print( [ importPath, modelSettings ] )
                result = translator.translate( translateMe, sourceLanguage=modelSettings[ 'sourceLanguage' ], targetLanguage=modelSettings[ 'targetLanguage' ] )
                assert( len( result ) == len( translateMe ) )
            except:
                print( 'Error translating data using the following settings:' )
                print( [ importPath, modelSettings ] )
                response = { }
                response[ 'translatedData' ] = None
                response[ 'engine' ] = modelSettings[ 'engine' ]
                response[ 'modelGroup' ] = modelSettings[ 'modelGroup' ]
                response[ 'modelName' ] = translator.modelName
                response[ 'device' ] = translator.device
                response[ 'hash' ] = translator.hash
                response[ 'processID' ] = os.getpid()
                translator = None
                outputQueue.put( response )
                raise
                return

        #returns { translatedData: [ translatedData ], engine: engine, modelGroup: modelGroup, modelName: modelName, device=activeDevice, hash : modelHash, processID : os.getpid() }
        response = { }
        response[ 'translatedData' ] = result
        response[ 'engine' ] = modelSettings[ 'engine' ]
        response[ 'modelGroup' ] = modelSettings[ 'modelGroup' ]
        response[ 'modelName' ] = translator.modelName
        response[ 'device' ] = translator.device
        response[ 'hash' ] = translator.hash
        response[ 'processID' ] = os.getpid()
        outputQueue.put( response )

#Added to check if inputQueue.put( job ) was blocking. Was not the issue. Issues was outputQueue needed to be proxied twice.
async def inputQueueGetProxy( executor, inputQueue, job ):
    inputQueue.put( job )
        #with concurrent.futures.ProcessPoolExecutor( max_workers=1, mp_context = multiprocessing.get_context( defaultProcessesSpawnTechnique ) ) as executor:
        #with concurrent.futures.ThreadPoolExecutor( ) as executor: #max_workers=2
            # executor, rawText, lock, modelDatabase, inputQueue, outputQueue, preloadModel, sourceLanguage, targetLanguage, engine, name, device
            #await asyncio.create_task( inputQueueGetProxy( executor, inputQueue, job ) )

def getFromOutputQueue( outputQueue ):
    return outputQueue.get()

# Added to check if outputQueue.get() was blocking. Solution was to proxy it twice. Just 'await asyncio.create_task()' blocked the main thread. Starting a new thread, and having that new thread schedule a callback using asyncio.get_running_loop().run_in_executor() finally fixed it.
# Probably what needs to happen is that the new task must be created in a new thread which is probably what asyncio.gather does. This sort of logic should probably be encapsulated somehow so the calling function can just do resolveInThread( functionName, { params } ) or resolveInProcess( functionName, { params } ) and get the data back without having to explicitly set up proxies.
# This logic below is what worked.
    #with concurrent.futures.ThreadPoolExecutor( ) as executor: #max_workers=2
        #taskList.append( asyncio.create_task( outputQueueGetProxy( executor, outputQueue ) ) )
        #outputText = await asyncio.gather( *taskList )
        #executor.shutdown( wait=False )
    #Remove the outer list.
    results = outputText[ 0 ]
async def outputQueueGetProxy( executor, outputQueue ):
    #return outputQueue.get()
    return await asyncio.get_running_loop().run_in_executor( executor, getFromOutputQueue, outputQueue )
    #return await asyncio.get_running_loop().run_in_executor(executor, preloadModelTranslate, rawText)
    #results = outputQueue.get() # Does this block? Update: This does not seem to be the issue, or there may be another one.
    #with concurrent.futures.ProcessPoolExecutor( max_workers=1, mp_context = multiprocessing.get_context( defaultProcessesSpawnTechnique ) ) as executor:
    #with concurrent.futures.ThreadPoolExecutor( ) as executor: #max_workers=2
        # executor, rawText, lock, modelDatabase, inputQueue, outputQueue, preloadModel, sourceLanguage, targetLanguage, engine, name, device
        #results = await asyncio.create_task( outputQueueGetProxy( executor, outputQueue ) )


#async def proxyTranslateNMT( executor, translateMe ):
    #return await asyncio.get_running_loop().run_in_executor( executor, translateNMT, translateMe )


# This function allows run_in_executor() to be added to a taskList, which is a Python list, and then awaiting the taskList.
# That will process all of the entries at once with an instance of concurrent.futures.ProcessPoolExecutor.
# Otherwise, each instance of each task will block the next and also maybe the ioloop depending upon implementation details.
#( rawText, the lock, modelDatabase dictionary, modelDatabaseValidated dictionary, inputQueue, outputQueue, preloadModel boolean, sourceLanguage, targetLanguage, engine, model, device ) # These last 5 are optional and may be None 
async def handleTranslationRequest( executor, rawText, lock, modelDatabase, modelDatabaseValidated, inputQueue, outputQueue, preloadModel, sourceLanguage=None, targetLanguage=None, engine=None, modelGroup=None, modelName=None, device=None ):
    debug = True
    if debug == True:
        print( 'handleTranslationRequest() input' )
        print( rawText, lock, preloadModel ) #inputQueue, outputQueue
        print( sourceLanguage, targetLanguage, engine, modelGroup, modelName, device)

    # Validate input.
    # Determine the chosenModule and chosenModel. Possible scenarios:
    #1. preloadModel == True. Solution: ignore input.
    #2. User specified engine, modelGroup, and modelName. Solution: Use engine.modelGroup using data fround in the modelName found in the [ modelGroup ][ 'models' ]. Error out or override as necessary. Error out if that modelName is not found as a key in in [ modelGroup ][ 'models' ].
    #3. User specified engine and modelGroup but not modelName. Solution: Use engine.modelGroup given by user with the default model, [ 'settings' ][ 'modelName' ] for that engine. Error out or override other options as necessary.
    #4. The user specified a modelName but not engine or modelGroup. In that case, search for the modelName in modelDatabaseValidated. If found, use the first modelname found.
    #5. Model is loaded, and user did not specify an engine, modelGroup, or modelName. User may have specified incomplete input like modelGroup or modelName, but not engine and vica-versa. Solution: Use loaded model respecting other specified options like sourgeLanguage, targetLanguage, and device.
    #6. Model is not loaded, and user did not specify a valid engine, modelGroup, and modelName. Solution: Get modelGroup to load from modelDatabaseVerified[ 'defaultModelGroup' ], load the default modelGroup, and load the default model for that group, model=modelDatabaseVerified[modelgroup][ 'settings' ][ 'modelName' ],  modelDatabaseVerified[ 'models' ][ model ]. Error out if that model does not work.

    chosenModule = None #engine + modelGroup
    chosenModel = None #modelName
    # Case 1.
    if preloadModel == True:
        if debug == True:
            print( 'Case1' )
        # Sanity check.
        assert( modelDatabase[ 'active' ] != None )
        chosenModule = modelDatabase[ 'active' ][ 'engine' ] + '.' + modelDatabase[ 'active' ][ 'modelGroup' ]
        chosenModel = modelDatabase[ 'active' ][ 'modelName' ]
    # Case 2.
    elif ( isinstance( engine, str ) == True ) and ( isinstance( modelGroup, str ) == True ):
        if debug == True:
            print( 'Case2' )
        if ( engine + '.' + modelGroup ) in modelDatabaseValidated:
            chosenModule = engine + '.' + modelGroup
            if isinstance( modelName, str ) == True:
                if modelName in modelDatabaseValidated[ chosenModule ][ 'models' ]:
                    chosenModel = modelName
            #else: # Case 3.
            if chosenModel == None:
                print( 'Case3' )
                chosenModel = modelDatabaseValidated[ chosenModule ][ 'settings' ][ 'modelName' ]
    # Case 4.
    elif isinstance( modelName, str ) == True:
        if debug == True:
            print( 'Case4' )
        for modulePath in modelDatabaseValidated:
            if modulePath == 'defaultModelGroup':
                continue
            for model in modelDatabaseValidated[ modulePath ][ 'models' ]:
                #if chosenModel == None:
                if model == modelName:
                    chosenModel = model
                    chosenModule = modulePath
                    break
    # Case 5.
    elif modelDatabase[ 'active' ] != None:
        if debug == True:
            print( 'Case5' )
        chosenModule = modelDatabase[ 'active' ][ 'engine' ] + '.' + modelDatabase[ 'active' ][ 'modelGroup' ]
        chosenModel = modelDatabase[ 'active' ][ 'modelName' ]
    # Case 6.
    if chosenModule == None:
        if debug == True:
            print( 'Case6' )
        chosenModule = modelDatabaseValidated[ 'defaultModelGroup' ]
        chosenModel = modelDatabaseValidated[ chosenModule ][ 'settings' ][ 'modelName' ]

    if debug == True:
        print( 'chosenModule', chosenModule )
        print( 'chosenModel', chosenModel )

    try:
        assert( ( chosenModel in modelDatabaseValidated[ chosenModule ][ 'models' ] ) == True )
    except:
        print( modelDatabase.keys() )
        print( modelDatabaseValidated.keys() )
        print( 'modelName in chosenModule: ', modelName in modelDatabaseValidated[ chosenModule ][ 'models' ] )
        print( engine )
        print( modelGroup )
        print( chosenModel )
        print( 'Unspecified error .' )
        raise

    # Try to create reponse from cache. Cache needs sourceLanguage and targetLanguage defined first.
    if sourceLanguage == None:
        sourceLanguage = modelDatabaseValidated[ chosenModule ][ 'models' ][ chosenModel ][ 'sourceLanguage' ]
    else:
        if ( sourceLanguage in modelDatabaseValidated[ chosenModule ][ 'models' ][ chosenModel ][ 'sourceLanguages' ] ) == False:
            sourceLanguage = modelDatabaseValidated[ chosenModule ][ 'models' ][ chosenModel ][ 'sourceLanguage' ]

    if targetLanguage == None:
        targetLanguage = modelDatabaseValidated[ chosenModule ][ 'models' ][ chosenModel ][ 'targetLanguage' ]
    else:
        if ( targetLanguage in modelDatabaseValidated[ chosenModule ][ 'models' ][ chosenModel ][ 'targetLanguages' ] ) == False:
            targetLanguage = modelDatabaseValidated[ chosenModule ][ 'models' ][ chosenModel ][ 'targetLanguage' ]

    # This should also be in modelDatabase[ chosenModule ][ 'models' ][ chosenModel ][ 'modelPath' ]
    modelPath = modelDatabase[ chosenModule ][ 'models' ][ chosenModel ][ 'modelPath' ]
    try:
        assert( modelPath != None )
    except:
        #print( modelDatabase )
        print()
        print( modelDatabaseValidated )
        print()
        print( chosenModule, chosenModel, modelPath )
        raise

    # Debug code.
    cacheEnabled = True
    #modelDatabase[ modulePath ][ 'models' ][ model ][ 'cache' ] = None
    #modelDatabase[ modulePath ][ 'models' ][ model ][ 'hash' ] = None

    if cacheEnabled != True:
        translateMe = rawText
        tempRequestList = [ ]
    elif cacheEnabled == True:
        # if needed, load cache.
        if modelDatabase[ chosenModule ][ 'models' ][ chosenModel ][ 'cache' ] == None:
            modelDatabase[ chosenModule ][ 'models' ][ chosenModel ][ 'cache' ] = multiLanguageCache.MultiLanguageCache( modelPath, modelFriendlyName=chosenModel, hash=modelDatabase[ chosenModule ][ 'models' ][ chosenModel ][ 'hash' ] )
            if modelDatabase[ chosenModule ][ 'models' ][ chosenModel ][ 'hash' ] == None:
                modelDatabase[ chosenModule ][ 'models' ][ chosenModel ][ 'hash' ] = modelDatabase[ chosenModule ][ 'models' ][ chosenModel ][ 'cache' ].hash
        if len( modelDatabase[ chosenModule ][ 'models' ][ chosenModel ][ 'cache' ].cache ) == 0:
            translateMe = rawText
            tempRequestList = [ ]
        else:
            # tempRequestList[ i ] = [ 'rawEntry', thisValueIsFromCache, translatedData ]
            translateMe, tempRequestList = await getFromCache( rawText, modelDatabase[ chosenModule ][ 'models' ][ chosenModel ][ 'cache' ], sourceLanguage, targetLanguage )
    if len( translateMe ) == 0:
        # All entries were found in the cache.
        finalList = [ ]
        for entry in tempRequestList:
            finalList.append( entry[ 2 ] )
        return finalList
    #else: # At least some entries need to be translated.

    # https://docs.python.org/3/library/asyncio-sync.html#asyncio.Lock
    print( 'lock status is ' + str( lock.locked() ) )
    #await lock.acquire()
    async with lock:

        # Debug code.
        #await asyncio.sleep( 3.14 )
        #print('cacheEnabled ',cacheEnabled)

        # Cache might have been updated enough to fulfill some or all of the subsequent queued request, so check it again here.
        if cacheEnabled != True:
            translateMe = rawText
            for i in translateMe:
                tempRequestList.append( (i, False) )
        elif cacheEnabled == True:
            # if needed, load cache.
            if modelDatabase[ chosenModule ][ 'models' ][ chosenModel ][ 'cache' ] == None:
                modelDatabase[ chosenModule ][ 'models' ][ chosenModel ][ 'cache' ] = multiLanguageCache.MultiLanguageCache( modelPath, modelPath, hash=modelDatabase[ chosenModule ][ 'models' ][ chosenModel ][ 'hash' ] )
                if modelDatabase[ chosenModule ][ 'models' ][ chosenModel ][ 'hash' ] == None:
                    modelDatabase[ chosenModule ][ 'models' ][ chosenModel ][ 'hash' ] = modelDatabase[ chosenModule ][ 'models' ][ chosenModel ][ 'cache' ].hash
            if len( modelDatabase[ chosenModule ][ 'models' ][ chosenModel ][ 'cache' ].cache ) == 0:
                translateMe = rawText
                for i in translateMe:
                    tempRequestList.append( ( i, False ) )
            else:
                # tempRequestList[ i ] = [ 'rawEntry', thisValueIsFromCache, translatedData ]
                translateMe, tempRequestList = await getFromCache( rawText, modelDatabase[ chosenModule ][ 'models' ][ chosenModel ][ 'cache' ], sourceLanguage, targetLanguage )
        if len( translateMe ) == 0:
            # All entries were found in the cache.
            finalList = [ ]
            for entry in tempRequestList:
                finalList.append( entry[ 2 ] )
            #lock.release()
            return finalList

        # if the user specified a device during their request, then that device must match the device the currently loaded model is using. How can that be validated? A: check modelDatabase[ 'active' ][ 'device' ], however preloadModel==True should ignore device as userInput since that was hard-set at program's launch time.
        if preloadModel == False:
            restartProcess = False
            if modelDatabase[ 'active' ] == None:
                # Then start the process.
                # This seems inconsistent to me. Why does translateNMT need its scope declared here? It likely has something to do with this function being called while inside of a class. At the same time, there is only one translateNMT function to begin with, the class does not have one, so what is causing the confusion or ambiguity? Other classes can use functions outside of their class without the scope getting declared, so why dpes this one need special treatment?
                global translateNMT
                multiprocessing.Process( target=translateNMT, args=( inputQueue, outputQueue ) ).start()
            elif ( device != None ) and ( device != modelDatabase[ 'active' ][ 'device' ] ):
                # Then the process must be restarted on the correct device.
                print( 'Restarting NMT process due to device mismatch.' )
                print( 'currentDevice: ' + modelDatabase[ 'active' ][ 'device' ] + '\nnewDevice: ' + device )
                restartProcess = True
            else:
                # The active module and model must also match.
                activeModule = modelDatabase[ 'active' ][ 'engine' ] + '.' + modelDatabase[ 'active' ][ 'modelGroup' ]
                activeModel = modelDatabase[ 'active' ][ 'modelName' ]
                if chosenModule != activeModule:
                    print( 'Restarting NMT process due to modelGroup mismatch.' )
                    print( 'chosenModule: ' + chosenModule + '\nactiveModule: ' + activeModule )
                    restartProcess = True
                elif activeModel != chosenModel:
                    print( 'Restarting NMT process due modelName mismatch.' )
                    print( 'activeModel: ' + activeModel + '\nchosenModel: ' + chosenModel )
                    restartProcess = True
            if restartProcess == True:
                inputQueue.put( 'stopProcess' )
                multiprocessing.Process( target=translateNMT, args=( inputQueue, outputQueue ) ).start()


        # Create job.
        # job = ( translateMe data, modelSettings={ engine, modelGroup, modelName, modelPath, device, sourceLanguage, targetLanguage, hash, tokenizer settings } )
        # Create requestDictionary. This will 1) be used to determine the translateNMT() process what to do, and 2) will be passed to the module/engineGroup verbatim. Any module-wide default settings must be updated to model specific settings here or the engine will recieve incorrect settings.
        #requestDictionary = modelDatabase[ chosenModule ][ 'models' ][ chosenModel ].copy()  # The module already has this information since that is where modelDatabase[ chosenModule ][ 'models' ] originally came from modelDatabase[ chosenModule ].models. There is no need to give it fully back. Just update the module defaults as needed.
        #Start by copying the module and all default settings, then update them.
        requestDictionary = modelDatabase[ chosenModule ][ 'settings' ].copy() #This already contains engine and modelGroup.
        if 'iniSettings' in modelDatabase[ chosenModule ][ 'models' ][ chosenModel ]:
            requestDictionary.update( modelDatabase[ chosenModule ][ 'models' ][ chosenModel ][ 'iniSettings' ] )
        requestDictionary[ 'modelName' ] = chosenModel
        requestDictionary[ 'modelPath' ] = modelPath
        # if the user specified a device during the request, use it, otherwise defer to the model's default device.
        if device != None:
            requestDictionary[ 'device' ] = device
        else:
            # The model's device.
            if ( 'device' in modelDatabase[ chosenModule ][ 'models' ][ chosenModel ] ) == True:
                requestDictionary[ 'device' ] = modelDatabase[ chosenModule ][ 'models' ][ chosenModel ][ 'device' ]
            else:
                # The modelGroup's default device.
                requestDictionary[ 'device' ] = modelDatabase[ chosenModule ][ 'settings' ][ 'device' ]
        requestDictionary[ 'sourceLanguage' ] = sourceLanguage
        requestDictionary[ 'targetLanguage' ] = targetLanguage
        requestDictionary[ 'hash' ] = modelDatabase[ chosenModule ][ 'models' ][ chosenModel ][ 'hash' ]

        job = ( translateMe, requestDictionary )

    #    try:
            # Submit job.
        inputQueue.put( job )
    #    except:
    #        print( 'aa' )
    #        inputQueue.put( 'stopProcess' )
    #        modelDatabase[ 'active' ] = None
    #        lock.release()
    #        return [ ]
    #        raise
        #with concurrent.futures.ProcessPoolExecutor( max_workers=1, mp_context = multiprocessing.get_context( defaultProcessesSpawnTechnique ) ) as executor:
        #with concurrent.futures.ThreadPoolExecutor( ) as executor: #max_workers=2
            # executor, rawText, lock, modelDatabase, inputQueue, outputQueue, preloadModel, sourceLanguage, targetLanguage, engine, name, device
            #await asyncio.create_task( inputQueueGetProxy( executor, inputQueue, job ) )

            #return await asyncio.get_running_loop().run_in_executor( executor, translateNMT, translateMe )


        # Handle result.
        #returns { translatedData: [ translatedData ], engine: engine, modelGroup: modelGroup, modelName: modelName, device=activeDevice, hash : modelHash, processID : os.getpid() }
        #results = outputQueue.get() # Does this block?
        taskList = [ ]
        #with concurrent.futures.ProcessPoolExecutor( max_workers=1, mp_context = multiprocessing.get_context( defaultProcessesSpawnTechnique ) ) as executor:
        with concurrent.futures.ThreadPoolExecutor( ) as executor: #max_workers=2
            # executor, rawText, lock, modelDatabase, inputQueue, outputQueue, preloadModel, sourceLanguage, targetLanguage, engine, name, device
            #results = await asyncio.create_task( outputQueueGetProxy( executor, outputQueue ) )
            taskList.append( asyncio.create_task( outputQueueGetProxy( executor, outputQueue ) ) )
            outputText = await asyncio.gather( *taskList )
            executor.shutdown( wait=False )
        #The above returns a list which encapsulates all 1 entries in the taskList. The preloadModelTranslate function itself also returns a list, so there is a [[]] object returned.
        #Remove the outer list.
        results = outputText[ 0 ]

        translatedResults = results[ 'translatedData' ]
        modelDatabase[ 'active' ] = { }
        modelDatabase[ 'active' ][ 'engine' ] = results[ 'engine' ]
        modelDatabase[ 'active' ][ 'modelGroup' ] = results[ 'modelGroup' ]
        modelDatabase[ 'active' ][ 'modelName' ] = results[ 'modelName' ]
        modelDatabase[ 'active' ][ 'device' ] = results[ 'device' ]
        modelDatabase[ 'active' ][ 'hash' ] = results[ 'hash' ]
        modelDatabase[ 'active' ][ 'processID' ] = results[ 'processID' ]
        modelDatabase[ 'active' ][ 'lastActiveTime' ] = time.time()

        if translatedResults == None:
            # Getting a None response for translatedResults means the process encounted an error and had to close using a return statement. It should not longer be active as long as it did not hang or encounter an uncaught exception, so just assume it closed correctly and forget all knowledge of it. This has the risk of creating zombie processes if they did not close correctly. They should still be registered correctly as child processes by the OS, so they will still close when KeyboardInterrupt is called by the final call to psutil to close all child processes forcefully.
            modelDatabase[ 'active' ] = None 
            #lock.release()
            return [ ]
        else:
            assert( len( translatedResults ) == len ( translateMe ) )

        if cacheEnabled != True:
            finalOutputList = translatedResults
        else:
            #merge tempRequestList and translatedResults to produce finalOutputList
            finalOutputList = [ ]
            # update cache whenever an untranslated entry is found
            translatedResultsPointer = 0
            for i in tempRequestList:
                if i[ 1 ] == True: #cache found...
                    finalOutputList.append( i[ 2 ] )
                else:
                    finalOutputList.append( translatedResults[ translatedResultsPointer ] )
                    # modelDatabase[ chosenModule ][ 'models' ][ chosenModel ][ 'cache' ].add( untranslatedString, translatedString, sourceLanguage, targetLanguage ) #TODO: Implement this.
                    translatedResultsPointer += 1

        #if lock.locked() == True:
        #    lock.release()

    # Debug code.
    #print( 'rawText',len(rawText), rawText )
    #print( 'translateMe',len(translateMe), translateMe )
    #print( 'tempRequestList ',len(tempRequestList), tempRequestList )
    #print( 'finalOutputList',len(finalOutputList), finalOutputList )
    assert( len( tempRequestList ) == len( rawText ) == len( finalOutputList ) )
    return finalOutputList


class MainHandler( tornado.web.RequestHandler ):
    # https://www.tornadoweb.org/en/stable/web.html#tornado.web.RequestHandler.initialize
    # initialize() allows MainHandler to be called as   (r'/', MainHandler, dict( abc='passed argument') ),
    #def initialize( self, abc=None ):
        # self.abc can be accessed from class member functions, like post() and get()
        #self.abc = abc
    def initialize( self, lock, modelDatabase, modelDatabaseValidated, inputQueue, outputQueue, preloadModel ):
        self.lock = lock
        self.modelDatabase = modelDatabase
        self.modelDatabaseValidated = modelDatabaseValidated
        self.inputQueue = inputQueue
        self.outputQueue = outputQueue
        self.preloadModel = preloadModel

    async def get( self ):
        print( 'self.request=' + str( self.request ) )
        #print( type(uiHTMLContents) )
        #if uiHTMLContents == None:
        #    return
        uiHTML = str( pathlib.Path( __file__ ).parent ) + '/resources/ui/ui.html'
        with open( uiHTML, 'rt', encoding='utf-8') as file:
            uiHTMLContents = file.read()
        self.write( uiHTMLContents )
        #self.render( 'resources/ui/ui.html' ) #Render is for use with tornado's HTML template system and is cached after it is generated. Not what I want.


    async def post(self):
        requestStartTime = time.perf_counter()
        print( 'self.request.body=' + str( self.request.body ) )

        self.set_header( 'Content-Type', 'application/json' ) # Set automatically by Tornado, so redundant.
        self.set_status( 200 )

        if self.request.headers.get( 'Content-Type') == 'application/json' :
            self.args = tornado.escape.json_decode( self.request.body )
        else:
            print( 'Error: Only json is supported as input currently. Returning.' )
            return
        if ( self.args == None ) or ( self.args == '' ):
            print( 'Error: No json contents found in request.body. Returning.' )
            return
        if isinstance( self.args, dict ) == False:
            print( 'Error: request.body did not return a Python dictionary. Returning.' )
            return

        rawText = None
        for key in self.args.keys():
            if 'content' == key.lower():
                rawText = self.args[ 'content' ]
        if rawText == None:
            print( 'No content to translate. Returning.' )
            return

        sourceLanguage=None
        for key in self.args.keys():
            # Convert everything to lower case to make requests case insensitive.
            if 'sourceLanguage'.lower() == key.lower():
                sourceLanguage = self.args[ key ].strip()
                if sourceLanguage == '':
                    sourceLanguage = None
                break

        targetLanguage=None
        for key in self.args.keys():
            if 'targetLanguage'.lower() == key.lower():
                targetLanguage = self.args[ key ].strip()
                if targetLanguage == '':
                    targetLanguage = None
                break

        engine=None
        for key in self.args.keys():
            if 'engine' == key.lower():
                engine = self.args[ key ].lower().strip()
                if engine == '':
                    engine = None
                break

        modelGroup=None
        for key in self.args.keys():
            if 'modelGroup'.lower() == key.lower():
                modelGroup = self.args[ key ].lower().strip()
                if modelGroup == '':
                    modelGroup = None
                break
        if modelGroup == None:
            for key in self.args.keys():
                if 'group' == key.lower():
                    modelGroup = self.args[ key ].lower().strip()
                    if modelGroup == '':
                        modelGroup = None
                    break

        modelName=None
        for key in self.args.keys():
            if 'modelName'.lower() == key.lower():
                modelName = self.args[ key ].lower().strip()
                if modelName == '':
                    modelName = None
                break
        if modelName==None:
            for key in self.args.keys():
                if 'model' == key.lower():
                    modelName = self.args[ key ].lower().strip()
                    if modelName == '':
                        modelName = None
                    break

        device=None
        for key in self.args.keys():
            if 'device' == key.lower():
                device = self.args[ key ].lower().strip()
                if device == '':
                    device = None
                break

        #print( 'rawText=',rawText )
        #print( 'type=', type( rawText ))
        if isinstance( rawText, str ):
            rawText = [ rawText ]
        #sys.exit()

        taskList = [ ]
        finalResults = [ ]
        # change this to ThreadPoolExecutor
        #with concurrent.futures.ProcessPoolExecutor( max_workers=1, mp_context = multiprocessing.get_context( defaultProcessesSpawnTechnique ) ) as executor:
        with concurrent.futures.ThreadPoolExecutor( ) as executor: #max_workers=2
            # executor, rawText, lock, modelDatabase, inputQueue, outputQueue, preloadModel, sourceLanguage, targetLanguage, engine, name, device
            finalResults = await asyncio.create_task( handleTranslationRequest( executor, rawText, self.lock, self.modelDatabase, self.modelDatabaseValidated, self.inputQueue, self.outputQueue, self.preloadModel, sourceLanguage, targetLanguage, engine, modelGroup, modelName, device ) )
            #taskList.append( asyncio.create_task( handleTranslationRequest( executor, rawText, self.lock ) ) )
            #for f in asyncio.as_completed( taskList ):
            #    finalResults.append( await f )

        #print( finalResults )
        #finalResults = finalResults[ 0 ]
        print( finalResults )
        #print( self.args[ 'content' ] )

        if isinstance( self.args[ 'content' ], str ) == True:
            finalResults = finalResults[ 0 ]

        print( 'Request servicing time: ' + str( round( time.perf_counter()  - requestStartTime, 2 ) )+ 's' )

        self.write( json.dumps( finalResults ) )


async def checkIdleProcess( modelDatabase, lock, unloadTime, inputQueue ):
    #print( 'Checking if process is idle' )
    # if there is an active model loaded and it is not locked,
    if ( modelDatabase[ 'active' ] != None ) and ( lock.locked() == False ):
        # then check if it has been idle for a while.
        #print( time.time() - modelDatabase[ 'active' ][ 'lastActiveTime' ], unloadTime, ( time.time() - modelDatabase[ 'active' ][ 'lastActiveTime' ] ) > unloadTime )
        if ( time.time() - modelDatabase[ 'active' ][ 'lastActiveTime' ] ) > unloadTime:
            print( 'Idle process found. Shutting it down.' )
            inputQueue.put( 'stopProcess' )
            modelDatabase[ 'active' ] = None

    #asyncio.get_running_loop().call_later( processTimout, checkIdleProcess, modelDatabase ) #this works for syncronous functions
    asyncio.get_event_loop().call_later( callbackResolution, asyncio.get_event_loop().create_task, checkIdleProcess( modelDatabase, lock, unloadTime, inputQueue ) )





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
    def initialize( self, scriptNameWithVersion ):
        self.scriptNameWithVersion=scriptNameWithVersion

    async def get( self ):
        print( 'self.request=' + str( self.request ) )
        if debug == True:
            print( 'Executing: ' + type( self ).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status( 200 )
        self.set_header( 'Content-Type', 'text/plain' )
        self.write( self.scriptNameWithVersion )

    async def post( self ):
        print( 'self.request=' + str( self.request ) )
        if debug == True:
            print( 'Executing: ' + type( self ).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status( 200 )
        self.set_header( 'Content-Type', 'application/json' )
        self.write( json.dumps( { 'content' : self.scriptNameWithVersion } ) )


class ReturnModel( tornado.web.RequestHandler ):
    def initialize( self, modelDatabase, modelDatabaseValidated ):
        self.modelDatabase=modelDatabase
        self.modelDatabaseValidated=modelDatabaseValidated

    async def get( self ):
        print( 'self.request=' + str( self.request ) )
        if debug == True:
            print( 'Executing: ' + type(self).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status( 200 )
        self.set_header( 'Content-Type', 'text/plain' )
        if self.modelDatabase[ 'active' ] != None:
            # The modelName, especially when paired with an engine, is already unique, so the hash does not matter much and makes the UI inconsistent whenever a model lacks a hash.
            response = self.modelDatabase[ 'active' ][ 'engine' ] + '/' + self.modelDatabase[ 'active' ][ 'modelName' ] # + self.modelDatabase[ 'active' ][ 'hash' ]
            self.write( response )
        else:
            defaultModulePath = self.modelDatabaseValidated[ 'defaultModelGroup' ]
            engine = self.modelDatabase[ defaultModulePath ][ 'engine' ]
            defaultModel = self.modelDatabaseValidated[ defaultModulePath ][ 'settings' ][ 'modelName' ]
            self.write( engine + '/' + defaultModel )

    async def post(self):
        print( 'self.request=' + str( self.request ) )
        if debug == True:
            print( 'Executing: ' + type( self ).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status( 200 )
        self.set_header( 'Content-Type', 'application/json' )

        if self.modelDatabase[ 'active' ] != None:
            response = self.modelDatabase[ 'active' ][ 'engine' ] + '/' + self.modelDatabase[ 'active' ][ 'modelName' ] # + self.modelDatabase[ 'hash' ]
            self.write( json.dumps( { 'content' : response } ) )
        else:
            defaultModulePath = self.modelDatabaseValidated[ 'defaultModelGroup' ]
            engine = self.modelDatabase[ defaultModulePath ][ 'engine' ]
            defaultModel = self.modelDatabaseValidated[ defaultModulePath ][ 'settings' ][ 'modelName' ]
            self.write( json.dumps( { 'content' : engine + '/' + defaultModel } ) )


class ReturnModelDatabase( tornado.web.RequestHandler ):
    def initialize( self, modelDatabase, modelDatabaseValidated ):
        self.modelDatabase=modelDatabase
        self.modelDatabaseValidated=modelDatabaseValidated.copy()

    async def get( self ):
        print( 'self.request=' + str( self.request ) )
        if debug == True:
            print( 'Executing: ' + type(self).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status( 200 )
        self.set_header( 'Content-Type', 'text/plain' )
        #modelDatabase[ 'active' ] = { }                                   # modelDatabase[ 'active' ] != None indicates an active model.
        #modelDatabase[ 'active' ][ 'engine' ] = None            # fairseq, ctranslate2, transformers
        #modelDatabase[ 'active' ][ 'modelGroup' ] = None  # sugoi, opusmt.opus-mt-tc-bible-big
        #modelDatabase[ 'active' ][ 'modelName' ] = None  # sugoi-v4, sugoi-levi, opus-mt-tc-bible-big-gmw-fra_ita_por_spa
        #modelDatabase[ 'active' ][ 'device' ] = None            # activeDevice
        #modelDatabase[ 'active' ][ 'hash' ] = None               # sha1hash
        if self.modelDatabase[ 'active' ] == None:
            self.modelDatabaseValidated[ 'active' ] = None
        else:
            tempDictionary = { }
            tempDictionary[ 'engine' ] = self.modelDatabase[ 'active' ][ 'engine' ]
            tempDictionary[ 'modelGroup' ] = self.modelDatabase[ 'active' ][ 'modelGroup' ]
            tempDictionary[ 'modelName' ] = self.modelDatabase[ 'active' ][ 'modelName' ]
            tempDictionary[ 'device' ] = self.modelDatabase[ 'active' ][ 'device' ]
            tempDictionary[ 'hash' ] = self.modelDatabase[ 'active' ][ 'hash' ]
            self.modelDatabaseValidated[ 'active' ] = tempDictionary
        self.write( str( self.modelDatabaseValidated ) )

    async def post(self):
        print( 'self.request=' + str( self.request ) )
        if debug == True:
            print( 'Executing: ' + type( self ).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status( 200 )
        self.set_header( 'Content-Type', 'application/json' )
        if self.modelDatabase[ 'active' ] == None:
            self.modelDatabaseValidated[ 'active' ] = None
        else:
            tempDictionary = { }
            tempDictionary[ 'engine' ] = self.modelDatabase[ 'active' ][ 'engine' ]
            tempDictionary[ 'modelGroup' ] = self.modelDatabase[ 'active' ][ 'modelGroup' ]
            tempDictionary[ 'modelName' ] = self.modelDatabase[ 'active' ][ 'modelName' ]
            tempDictionary[ 'device' ] = self.modelDatabase[ 'active' ][ 'device' ]
            tempDictionary[ 'hash' ] = self.modelDatabase[ 'active' ][ 'hash' ]
            self.modelDatabaseValidated[ 'active' ] = tempDictionary
        self.write( json.dumps( self.modelDatabaseValidated ) )


class ReturnRawDatabase( tornado.web.RequestHandler ):
    def initialize( self, modelsTrimmed ):
        self.modelsTrimmed=modelsTrimmed

    async def get( self ):
        print( 'self.request=' + str( self.request ) )
        if debug == True:
            print( 'Executing: ' + type(self).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status( 200 )
        self.set_header( 'Content-Type', 'text/plain' )
        self.write( str( self.modelsTrimmed ) )

    async def post(self):
        print( 'self.request=' + str( self.request ) )
        if debug == True:
            print( 'Executing: ' + type( self ).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        self.set_status( 200 )
        self.set_header( 'Content-Type', 'application/json' )
        self.write( json.dumps( self.modelsTrimmed ) )


class Search( tornado.web.RequestHandler ):
    def initialize( self, modelDatabase, modelDatabaseValidated ):
        #self.modelDatabase=modelDatabase
        #self.modelDatabaseValidated=modelDatabaseValidated.copy()
        pass

    async def get( self ):
        print( 'self.request=' + str( self.request ) )
        if debug == True:
            print( 'Executing: ' + type(self).__name__ + '.' + inspect.currentframe().f_code.co_name ) #Print out className.currentFunctionName.
        #self.set_status( 200 )
        #self.set_header( 'Content-Type', 'text/plain' )
        searchHTML = str( pathlib.Path( __file__ ).parent ) + '/resources/ui/search.html'
        with open( searchHTML, 'rt', encoding='utf-8') as file:
            searchHTMLContents = file.read()
        self.write( searchHTMLContents )


async def main():
    startedLoadingTime = time.perf_counter()

    # Set some generic defaults that need to be after the import statments.
    currentScriptPathObject = pathlib.Path( __file__ ).resolve()
    currentScriptNameWithoutPath = currentScriptPathObject.name
    currentScriptNameWithoutPathOrExt = currentScriptPathObject.stem
    usageHelp = ' Usage: ' + currentScriptNameWithoutPath + ' -h'
    # Set path of current script.
    currentScriptPathOnly = str( currentScriptPathObject.parent ) #Does not include last / and this will return one subfolder up if it is called on a folder.
    # Debug code.
    # absoluteSearchPathToEngines = currentScriptPathOnly + '/' + relativePathToEnginesFolderFromProgramRoot
    # absoluteSearchPathToEngines = str( pathlib.Path( __file__ ).parent.parent.parent )
    # programRoot = absoluteSearchPathToEngines #TODO: This should be __file__.parent only. currentScriptPathOnly should == programRoot.
    # programRoot= currentScriptPathOnly
    # absoluteSearchPathToEngines = programRoot + '/' + relativePathToEnginesFolderFromProgramRoot #TODO: programRoot should be currentScriptPathOnly

    scriptNameWithVersion = currentScriptNameWithoutPathOrExt + '/' +__version__
    #scriptNameWithVersionDictionary = { 'content' : scriptNameWithVersion }
    #modeAndModelName = engine + '/' + inputModelNameWithoutPath + '/' + modelHash
    #model[ 'active' ][ 'engine' ]
    #model[ 'active' ][ 'modelName' ]
    #model[ 'active' ][ 'hash' ]

    print( ( currentScriptNameWithoutPathOrExt + ' loading...' ).encode( consoleEncoding ) )

    programSettings = { }
    programSettings[ currentScriptNameWithoutPathOrExt ] = { }
    programSettings[ currentScriptNameWithoutPathOrExt ][ 'modelDatabase' ] = [ 'fairseq.sugoi', 'ctranslate2.sugoi', 'transformers.opusmt.opus-mt-tc-bible-big' ]
    programSettings[ currentScriptNameWithoutPathOrExt ][ 'device' ] = None

    # Initialize database.
    modelDatabase = { }

    # Start filling in database.
    if ( 'defaultModelGroup' in programSettings ) == True:
        modelDatabase[ 'defaultModelGroup' ] = programSettings[ 'defaultModelGroup' ]
    if ( 'defaultModelGroup' in modelDatabase ) == False:
        modelDatabase[ 'defaultModelGroup' ] = None # This is a slight misnomer. defaultModelGroup should probably be defaultModulePath, but it gets set in the .ini which is user-facing. In that context, defaultModelGroup is clearer since the .ini is almost entirely about configuring different modelGroups. Since that is user facing, that has priority here for variable naming despite the misnomer.

    # modulePath == engine.modelGroup as in 'fairseq.sugoi', 'ctranslate2.sugoi', or 'transformers.opusmt'
    for modulePath in programSettings[ currentScriptNameWithoutPathOrExt ][ 'modelDatabase' ]:
        modelDatabase[ modulePath ] = { }                             # This dictionary stores all information for the modelGroup.py
        modelDatabase[ modulePath ][ 'engine' ] = modulePath.split( '.', maxsplit=1 )[ 0 ].strip()
        modelDatabase[ modulePath ][ 'modelGroup' ] = modulePath.split( '.', maxsplit=1 )[ 1 ].strip()
        modelDatabase[ modulePath ][ 'imported' ] = False # This tracks if the module was successfully imported or not.
        modelDatabase[ modulePath ][ 'module' ] = None    # The engine library imported using importlib from which to extract variables.
        modelDatabase[ modulePath ][ 'models' ] = None    # A dictionary of model names imported from the engine library with model unique keys for modelPath, sourceLanguages list, targetLanguages list, modelURLs, and description.

        if modulePath in programSettings:
            # modelDatabase[ modulePath ][ 'settings' ] stores all default settings for the model group including the default model name to load when no specific model is selected, the device to load it on, and which languages to use. The settings here will be used whenever loading a model in the group if a particular request does not specify the exact settings to use.
            modelDatabase[ modulePath ][ 'settings' ] = programSettings[ modulePath ]
            if ( 'modelName' in modelDatabase[ modulePath ][ 'settings' ] ) == False:
                modelDatabase[ modulePath ][ 'settings' ][ 'modelName' ] = None
            if ( 'modelPath' in modelDatabase[ modulePath ][ 'settings' ] ) == False:
                modelDatabase[ modulePath ][ 'settings' ][ 'modelPath' ] = None
            if ( 'sourceLanguage' in modelDatabase[ modulePath ][ 'settings' ] ) == False:
                modelDatabase[ modulePath ][ 'settings' ][ 'sourceLanguage' ] = None
            if ( 'targetLanguage' in modelDatabase[ modulePath ][ 'settings' ] ) == False:
                modelDatabase[ modulePath ][ 'settings' ][ 'targetLanguage' ] = None
            if ( 'device' in modelDatabase[ modulePath ][ 'settings' ] ) == False:
                modelDatabase[ modulePath ][ 'settings' ][ 'device' ] = programSettings[ currentScriptNameWithoutPathOrExt ][ 'device' ]
            if ( 'devices' in modelDatabase[ modulePath ][ 'settings' ] ) == False:
                modelDatabase[ modulePath ][ 'settings' ][ 'devices' ] = None
        else:
            modelDatabase[ modulePath ][ 'settings' ] = { }
            modelDatabase[ modulePath ][ 'settings' ][ 'modelName' ] = None
            modelDatabase[ modulePath ][ 'settings' ][ 'modelPath' ] = None
            modelDatabase[ modulePath ][ 'settings' ][ 'sourceLanguage' ] = None
            modelDatabase[ modulePath ][ 'settings' ][ 'targetLanguage' ] = None
            modelDatabase[ modulePath ][ 'settings' ][ 'device' ] = programSettings[ currentScriptNameWithoutPathOrExt ][ 'device' ]
            modelDatabase[ modulePath ][ 'settings' ][ 'devices' ] = None
        # Duplicate this information into [ modulePath ][ 'settings' ] for convenience.
        modelDatabase[ modulePath ][ 'settings' ][ 'engine' ] = modelDatabase[ modulePath ][ 'engine' ]
        modelDatabase[ modulePath ][ 'settings' ][ 'modelGroup' ] = modelDatabase[ modulePath ][ 'modelGroup' ]

    # Add any missing engines based upon the presence of the raw.py files in the appropriate directories.
    # This returns a list of dictionaries.
    #def getEngines( programRoot, relativePathToEnginesFolderFromProgramRoot ):
    engines = getEngines( currentScriptPathOnly, relativePathToEnginesFolderFromProgramRoot )
    for engine in engines:
        if isinstance( engine, dict ) == False:
            continue
        modulePath = engine[ 'engine' ] + '.' + engine[ 'modelGroup' ]
        # if the modulePath already exists, then skip it.
        if modulePath in modelDatabase:
            continue
        modelDatabase[ modulePath ] = { }
        modelDatabase[ modulePath ][ 'engine' ] = engine[ 'engine' ]
        modelDatabase[ modulePath ][ 'modelGroup' ] = engine[ 'modelGroup' ]
        modelDatabase[ modulePath ][ 'imported' ] = False # This tracks if the module was successfully imported or not.
        modelDatabase[ modulePath ][ 'module' ] = None    # The engine library imported using importlib.
        modelDatabase[ modulePath ][ 'models' ] = None    # A dictionary of model names imported from the engine library with model unique keys for modelPath, sourceLanguages list, targetLanguages list, and modelURLs.

        modelDatabase[ modulePath ][ 'settings' ] = { }
        modelDatabase[ modulePath ][ 'settings' ][ 'engine' ] = modelDatabase[ modulePath ][ 'engine' ]
        modelDatabase[ modulePath ][ 'settings' ][ 'modelGroup' ] = modelDatabase[ modulePath ][ 'modelGroup' ]
        modelDatabase[ modulePath ][ 'settings' ][ 'modelName' ] = None
        modelDatabase[ modulePath ][ 'settings' ][ 'modelPath' ] = None
        modelDatabase[ modulePath ][ 'settings' ][ 'sourceLanguage' ] = None
        modelDatabase[ modulePath ][ 'settings' ][ 'targetLanguage' ] = None
        modelDatabase[ modulePath ][ 'settings' ][ 'device' ] = programSettings[ currentScriptNameWithoutPathOrExt ][ 'device' ]
        modelDatabase[ modulePath ][ 'settings' ][ 'devices' ] = None

    allModelsRaw = { } # This has modulePath mappings to all of the known models from all .py files, including modelPath/urls/descriptions, but not validated to see if they are present or not.
    #################################################################################
    sys.path.append( str( pathlib.Path( __file__ ).resolve().parent.parent.parent ) )
    for modulePath in modelDatabase:
        if modulePath == 'defaultModelGroup':
            continue
        try:
            modelDatabase[ modulePath ][ 'module' ] = importlib.import_module( relativePathToEnginesFolderFromProgramRoot.replace( '/', '.' ) + '.' + modelDatabase[ modulePath ][ 'engine' ] + '.' + modelDatabase[ modulePath ][ 'modelGroup' ] )
            modelDatabase[ modulePath ][ 'imported' ] = True
            modelDatabase[ modulePath ][ 'models' ] = modelDatabase[ modulePath ][ 'module' ].models

            allModelsRaw[ modulePath ] = { }
            #allModelsRaw[ modulePath ][ 'models' ] = { }
            allModelsRaw[ modulePath ][ 'models' ] = modelDatabase[ modulePath ][ 'models' ]#.copy() # The next line modifies allModelsRaw[ modulePath ] which actually points to modelDatabase[ modulePath ][ 'models' ], so that means modelDatabase[ modulePath ][ 'models' ] gets modified which messes up the original datastructure. To prevent that, create a .copy() or offset it as allModelsRaw[ modulePath ][ 'models' ].
            allModelsRaw[ modulePath ][ 'devices' ] = modelDatabase[ modulePath ][ 'module' ].devices
            #print( modelDatabase[ modulePath ][ 'module' ].devices )

            # if default model, modelName, is missing in config.ini, set default model based on modelGroup.py settings.
            if modelDatabase[ modulePath ][ 'settings' ][ 'modelName' ] == None:
                modelDatabase[ modulePath ][ 'settings' ][ 'modelName' ] = modelDatabase[ modulePath ][ 'models' ][ 'defaultModel' ]

            # if any required modulePath settings are missing, update them based upon the defaultModel settings from modelGroup.py
            # This will also work to update the settings if only modelName was found in config.ini [ engine.modelGroup ], but the modulePath entry was missing modelPath, sourceLanguage, or targetLanguage.
            defaultModelForGroup = modelDatabase[ modulePath ][ 'settings' ][ 'modelName' ]
            if modelDatabase[ modulePath ][ 'settings' ][ 'modelPath' ] == None:
                modelDatabase[ modulePath ][ 'settings' ][ 'modelPath' ] = modelDatabase[ modulePath ][ 'models' ][ defaultModelForGroup ][ 'modelPath' ]
            if modelDatabase[ modulePath ][ 'settings' ][ 'sourceLanguage' ] == None:
                modelDatabase[ modulePath ][ 'settings' ][ 'sourceLanguage' ] = modelDatabase[ modulePath ][ 'module' ].defaultSourceLanguage
                #modelDatabase[ modulePath ][ 'settings' ][ 'sourceLanguage' ] = modelDatabase[ modulePath ][ 'models' ][ defaultModelForGroup ][ 'sourceLanguages' ][ 0 ]
            if modelDatabase[ modulePath ][ 'settings' ][ 'targetLanguage' ] == None:
                modelDatabase[ modulePath ][ 'settings' ][ 'targetLanguage' ] = modelDatabase[ modulePath ][ 'module' ].defaultTargetLanguage
                #modelDatabase[ modulePath ][ 'settings' ][ 'targetLanguage' ] = modelDatabase[ modulePath ][ 'models' ][ defaultModelForGroup ][ 'targetLanguages' ][ 0 ]
            if modelDatabase[ modulePath ][ 'settings' ][ 'devices' ] == None:
                modelDatabase[ modulePath ][ 'settings' ][ 'devices' ] = modelDatabase[ modulePath ][ 'module' ].devices
        except:
            modelDatabase[ modulePath ][ 'imported' ] = False
            #raise

    # Now that all of of the default engines + user input engines + dynamically detected modules/modelGroups have been added to modelDatabase and imported, initialize cache and hash for each model.
    #modelDatabase[ modulePath ][ 'models' ][ model ][ 'cache' ] = None        # An instance of multiLanguageCache.MultiLanguageCache()
    #modelDatabase[ modulePath ][ 'models' ][ model ][ 'hash' ] = None         # The sha1 hash of the model.
    #print(modelDatabase)
    for modulePath in modelDatabase:
        if modulePath == 'defaultModelGroup':
            continue
        # Ignore any modelGroups that were not imported successfully.
        if modelDatabase[ modulePath ][ 'imported' ] == False:
            continue
        #print( modelDatabase[ modulePath ][ 'models' ] )
        #print( modelDatabase[ modulePath ][ 'models' ].keys() )

    # It is possible for the user to specify a default modelName as modelDatabase[ modulePath ][ 'settings' ][ 'modelName' ] and default modelPath as modelDatabase[ modulePath ][ 'settings'][ 'modelPath' ]. If they do this, then the modelPath in modelDatabase[ modulePath ][ 'models' ][ model ][ 'modelPath' ]= could be None. That key gets obtained as modelDatabase[ modulePath ][ 'models' ] = modelDatabase[ modulePath ][ 'module' ].models, and was never updated. A valid 'modelPath' value in at modelDatabase[ modulePath ][ 'models' ][ model ][ 'modelPath' ] is required later to initialize cache even when preloading the model or otherwise. So, cascade the defaults appropriately.
    for modulePath in modelDatabase:
        if modulePath == 'defaultModelGroup':
            continue
        # Ignore any modelGroups that were not imported successfully.
        if modelDatabase[ modulePath ][ 'imported' ] == False:
            continue
        defaultModelName = modelDatabase[ modulePath ][ 'settings' ][ 'modelName' ]
        defaultModelPath = modelDatabase[ modulePath ][ 'settings'][ 'modelPath' ]
        if ( isinstance( defaultModelName , str ) == True ) and ( isinstance( defaultModelPath, str ) == True ):
            if modelDatabase[ modulePath ][ 'models' ][ defaultModelName ][ 'modelPath' ] == None:
                modelDatabase[ modulePath ][ 'models' ][ defaultModelName ][ 'modelPath' ] = defaultModelPath

    # Allow user input for modelPath, device, sourceLanguage, targetLanguage, hash, and misc tokenizer settings for models in the .ini. [modulePath#modelName] and possibily the CLI.
    # Should be available as modelDatabase[ modulePath ][ 'models' ][ model ][ 'hash' ]
        for model in modelDatabase[ modulePath ][ 'models' ].keys():
            if model == 'defaultModel':
                continue
            try:
                modelDatabase[ modulePath ][ 'models' ][ model ][ 'cache' ] = None
            except:
                print( modelDatabase[ modulePath ] )
                print( 'modulePath', modulePath )
                print( 'model', model )
                print( modelDatabase[ modulePath ][ 'models' ][ model ] )
                print( modelDatabase[ modulePath ][ 'models' ][ model ][ 'cache' ] )
                raise
            modelDatabase[ modulePath ][ 'models' ][ model ][ 'hash' ] = None
            if ( 'sourceLanguage' in modelDatabase[ modulePath ][ 'models' ][ model ] ) == False:
                modelDatabase[ modulePath ][ 'models' ][ model ][ 'sourceLanguage' ] = None
            if ( 'targetLanguage' in modelDatabase[ modulePath ][ 'models' ][ model ] ) == False:
                modelDatabase[ modulePath ][ 'models' ][ model ][ 'targetLanguage' ] = None

    #if 'iniSettings' in modelDatabase[ modulePath ][ 'models' ][ chosenModel ]:
    # Now that all of the settings have been updated, incorporate any model-specific overrides found in the .ini.
    # There may be model specific settings provided by the user in the .ini in the form [modulePath#modelName]
    # The logical place to put these is modelDatabase[ modulePath ][ 'models' ][ model ][ 'settings' ], but that has a chance of being confused with modelDatabase[ modulePath ][ 'settings' ], modelDatabaseValidated[ modulePath ][ 'settings' ], and moduleDatabase[ modulePath ][ 'models' ][ model ], so rename it to modelDatabase[ modulePath ][ 'models' ][ model ][ 'iniSettings' ] to make their origin and purpose explicit.
    for heading in programSettings:
        if heading.find( '/' ) == -1:
            continue
        modulePath = heading.partition( '/' )[ 0 ].strip()
        model = heading.partition( '/' )[ 2 ].strip()
        if modulePath in modelDatabase:
            if model in modelDatabase[ modulePath ][ 'models' ]:
                modelDatabase[ modulePath ][ 'models' ][ model ][ 'iniSettings' ] = programSettings[ heading ]

    modelDatabase[ 'active' ] = None
    # When a model is currently loaded into memory, the modelDatabase should also have
    #engine: engine, modelGroup: modelGroup, modelName: modelName, hash : modelHash, device=activeDevice, processID : os.getpid()
    #modelDatabase[ 'active' ] = { }                                   # modelDatabase[ 'active' ] != None indicates an active model.
    #modelDatabase[ 'active' ][ 'engine' ] = None            # fairseq, ctranslate2, transformers
    #modelDatabase[ 'active' ][ 'modelGroup' ] = None  # sugoi, opusmt.opus-mt-tc-bible-big
    #modelDatabase[ 'active' ][ 'modelName' ] = None  # sugoi-v4, sugoi-levi, opus-mt-tc-bible-big-gmw-fra_ita_por_spa
    #modelDatabase[ 'active' ][ 'device' ] = None            # activeDevice
    #modelDatabase[ 'active' ][ 'hash' ] = None               # sha1hash
    #modelDatabase[ 'active' ][ 'processID' ] = None      # process ID of the child process that is currently running the model obtained using os.getpid()
    #modelDatabase[ 'active' ][ 'lastActiveTime' ] = None #time.time()

    #################################################################################
    # Debug code.
    chosenModulePath = 'ctranslate2.sugoi'
    modelNameTemp = 'sugoi-v4'
    ctranslate2SugoiModelPath = r'C:\Users\Public\Downloads\models\sugoi_v4model_ctranslate2'
    modelDatabase[ 'ctranslate2.sugoi' ][ 'settings' ][ 'modelName' ] = modelNameTemp    # This sets the default model to use for the module.
    modelDatabase[ 'ctranslate2.sugoi' ][ 'settings' ][ 'modelPath' ] = ctranslate2SugoiModelPath #This sets the path to use for the default modelName for the module.
    modelDatabase[ chosenModulePath ][ 'models' ][ modelNameTemp ][ 'modelPath' ] = ctranslate2SugoiModelPath # Update the modelPath for the default model.
    fairseqSugoiModelPath = r'C:\Users\Public\Downloads\models\sugoi_v4model_fairseq'
    modelDatabase[ 'fairseq.sugoi' ][ 'settings' ][ 'modelName' ] = modelNameTemp
    modelDatabase[ 'fairseq.sugoi' ][ 'settings' ][ 'modelPath' ] = fairseqSugoiModelPath
    modelDatabase[ 'fairseq.sugoi' ][ 'models' ][ modelNameTemp ][ 'modelPath' ] = fairseqSugoiModelPath


    #modelDatabase[ 'ctranslate2.sugoi' ][ 'settings' ][ 'modelName' ] # This has the default settings for the group, the modelName of the default model.
    #modelDatabase[ 'ctranslate2.sugoi' ][ 'settings' ][ 'modelPath' ] # This has the default settings for the group, the modelPath to the default model.
    #modelDatabase[ modulePath ][ 'models' ][ model ][ 'modelPath' ] #This is the path for a particular model.
    assert( pathlib.Path( modelDatabase[ 'ctranslate2.sugoi' ][ 'settings' ][ 'modelPath' ] ).exists() == True )

    # allModelsRaw is a mapping of { modulePath: module.models } but module.models has an extra 'devices' entry from module.devices. trimAllModelsRaw removes modelPath and the tokenizer settings, returning only generic non-sensitive values about all possible models that can be loaded.
    #print( allModelsRaw )
    modelsTrimmed = trimAllModelsRaw( allModelsRaw )
    #print( modelsTrimmed )
    #sys.exit( 0 )
    if debug == True:
        for key,item in modelsTrimmed.items():
            print( key, len( modelsTrimmed[ key ] ) )
            if len( modelsTrimmed[ key ] ) < 20:
                print( key, modelsTrimmed[ key ] )

    # validateModels makes sure every model exists, fixes paths by making sure modelPath that points to folders get fixed to point to .bin files if possible, and returns a summary of modelDatabase[ modelGroup ][ 'models' ] where each model entry has been validated and lacks sensitive or internal information like modelPath and tokenizer settings.
    #modelDatabaseValidated = { }
    #modelDatabaseValidated[ 'defaultModelGroup' ] = None
    #modelDatabaseValidated[ modelGroup] = { }
    #modelDatabaseValidated[ modelGroup ][ 'settings' ][ 'modelName' ] =       # The default model for the group.
    #modelDatabaseValidated[ modelGroup][ 'settings' ][ sourceLanguage ]   # The default sourceLanguage for the group.
    #modelDatabaseValidated[ modelGroup][ 'settings' ][ targetLanguage ]    # The default targetLanguage for the group.
    #modelDatabaseValidated[ modelGroup ][ 'models' ]= { }           # Each entry is a valid model and has these keys: sourceLanguages, targetLanguages, modelURLs, description.
    #print( modelDatabase )
    modelDatabaseValidated = validateModels( modelDatabase )
    #print( modelDatabaseValidated )
    if debug == True:
        print( modelDatabaseValidated )
    #print( )

    if len( modelDatabaseValidated ) == 0:
        print( 'Error: Unable to find any models to serve. Please check the modelPath settings and try again.' )
        print( usageHelp )
        sys.exit( 1 )

    #print( modelDatabase )
    print( '\nThe following models are available:' )
    print( '[modelGroup], [modelName], [defaultSourceLanguage] => [defaultTargetLanguage]' )
    counter = 0
    for modulePath in modelDatabaseValidated:
        if modulePath == 'defaultModelGroup':
            continue
        for model in modelDatabaseValidated[ modulePath ][ 'models' ]:
            counter += 1
            print( str( counter ) + '.', modulePath + ',', model + ',', modelDatabaseValidated[ modulePath ][ 'models' ][ model ][ 'sourceLanguages' ][ 0 ]+' => ' + modelDatabaseValidated[ modulePath ][ 'models' ][ model ][ 'targetLanguages' ][ 0 ] )
        #print( 'The default model for the', modulePath,'modulePath is', modelDatabaseValidated[ modulePath ][ 'settings' ][ 'modelName' ])
        #print( str( counter ) + ')', modulePath, modelDatabaseValidated[ modulePath ][ 'settings' ][ 'modelName' ], 'source=' + modelDatabaseValidated[ modulePath ][ 'settings' ][ 'sourceLanguage' ], 'target=' + modelDatabaseValidated[ modulePath ][ 'settings' ][ 'targetLanguage' ] )
        #print( str( counter ) + '.', modulePath + ',', modelDatabaseValidated[ modulePath ][ 'settings' ][ 'modelName' ] + ',', modelDatabaseValidated[ modulePath ][ 'settings' ][ 'sourceLanguage' ]+' => ' + modelDatabaseValidated[ modulePath ][ 'settings' ][ 'targetLanguage' ] )

    print( '\nThe default modelGroup is ' + modelDatabaseValidated[ 'defaultModelGroup' ] )
    #counter = 0
    for modulePath in modelDatabaseValidated:
        if modulePath == 'defaultModelGroup':
            continue
        #for model in modelDatabaseValidated[ modulePath ][ 'models' ]:
            #counter += 1
            #print( str( counter ) + '.', modulePath + ',', model + ',', modelDatabaseValidated[ modulePath ][ 'models' ][ model ][ 'sourceLanguages' ][ 0 ]+' => ' + modelDatabaseValidated[ modulePath ][ 'models' ][ model ][ 'targetLanguages' ][ 0 ] )
        print( 'The default model for', modulePath,'is', modelDatabaseValidated[ modulePath ][ 'settings' ][ 'modelName' ] )
        #print( str( counter ) + ')', modulePath, modelDatabaseValidated[ modulePath ][ 'settings' ][ 'modelName' ], 'source=' + modelDatabaseValidated[ modulePath ][ 'settings' ][ 'sourceLanguage' ], 'target=' + modelDatabaseValidated[ modulePath ][ 'settings' ][ 'targetLanguage' ] )
        #print( str( counter ) + '.', modulePath + ',', modelDatabaseValidated[ modulePath ][ 'settings' ][ 'modelName' ] + ',', modelDatabaseValidated[ modulePath ][ 'settings' ][ 'sourceLanguage' ]+' => ' + modelDatabaseValidated[ modulePath ][ 'settings' ][ 'targetLanguage' ] )
    print()

    #debug = True
    if debug == True:
        print( modelDatabase )
        print()
        for modulePath in modelDatabase:
            if ( modulePath == 'defaultModelGroup' ) or ( modulePath == 'active' ):
                continue
            if modelDatabase[ modulePath ][ 'imported' ] == False:
                continue
            print( 'Default model for '+ modulePath + ': ' + str( modelDatabase[ modulePath ][ 'settings' ][ 'modelPath' ] ) )
            for model in modelDatabase[ modulePath ][ 'models' ]:
                if model == 'defaultModel':
                    continue
                print( model +' '+ str( modelDatabase[ modulePath ][ 'models' ][ model ][ 'modelPath' ] ) )
    #sys.exit( 0 )

    # Now that input has been imported and validated, set more core internal variables.
    # Only one thread can process its data at any one time, so create a lock that is read by the various threads to tell them to wait until the process is free.
    #locked = [ False ]
    lock = asyncio.Lock()

    # These queues, that use .put() and .get(), are used so the main parent process can communicate translation job i/o with the child process.
    inputQueue = multiprocessing.Queue()
    outputQueue = multiprocessing.Queue()

    #Debug code.
    preloadModel = False
    #preloadModel = True

    if preloadModel == True:
        # Create process stub.
        multiprocessing.Process( target=translateNMT, args=( inputQueue, outputQueue ) ).start()

        # Create dummy job to initialize the model. The settings for the model should be obtained from parsing settings. Otherwise, use the data from modelDatabaseValidated.
        # job = ( translateMe data, modelSettings={ engine, modelGroup, modelName, modelPath, device, sourceLanguage, targetLanguage, hash, tokenizer settings } )
        # The inputs here should be obtained from preloadModel specific defaults and are all required to be valid together. For debugging, hardcode some settings.
        #chosenModulePath = 'ctranslate2.sugoi'
        #modelNameTemp = 'sugoi-v4'
        translateMe = [ 'dummyRequestForPreloadModel' ]
        requestDictionary = modelDatabase[ chosenModulePath ][ 'settings' ].copy()
        # if there is no default modelDatabase[ chosenModulePath ][ 'settings' ]['modelPath'], then a modelPath will need to be added to the request manually from modelDatabase[ chosenModulePath ][ 'models' ][ chosenModel ][ 'modelPath' ].
        if ( 'modelPath' in requestDictionary ) == False:
            requestDictionary[ 'modelPath' ] = modelDatabase[ chosenModulePath ][ 'models' ][ modelNameTemp ][ 'modelPath' ]
        #requestDictionary[ 'modelPath' ] = modelDatabase[ chosenModulePath ][ 'modelPath' ]
        job = ( translateMe, requestDictionary )

        # Submit job.
        inputQueue.put( job )

        # Handle result.
        #returns { translatedData: [ translatedData ], engine: engine, modelGroup: modelGroup, modelName: modelName, device=activeDevice, hash : modelHash, processID : os.getpid() }
        results = outputQueue.get()
        #translatedResults = results[ 'translatedData' ]
        modelDatabase[ 'active' ] = { }
        modelDatabase[ 'active' ][ 'engine' ] = results[ 'engine' ]
        modelDatabase[ 'active' ][ 'modelGroup' ] = results[ 'modelGroup' ]
        modelDatabase[ 'active' ][ 'modelName' ] = results[ 'modelName' ]
        modelDatabase[ 'active' ][ 'device' ] = results[ 'device' ]
        modelDatabase[ 'active' ][ 'hash' ] = results[ 'hash' ]
        modelDatabase[ 'active' ][ 'processID' ] = results[ 'processID' ]
        modelDatabase[ 'active' ][ 'lastActiveTime' ] = time.time()

    # UI
#    uiHTML = currentScriptPathOnly + '/resources/ui/ui.html'
#    global uiHTMLContents
#    try:
#    with open( uiHTML, 'rt', encoding='utf-8') as file:
#        uiHTMLContents = file.read()
#    except:
#        uiHTMLContents = None
    #print(uiHTMLContents)

    serverSettings = {
        'static_path' : currentScriptPathOnly + '/resources/ui',
        #'static_path' : os.path.join( os.path.dirname( __file__ ), 'resources/ui' ),
        'cookie_secret': str( random.randint(2**64,2**200) ),
        'xsrf_cookies': True,
    }
    class staticFileHandler( tornado.web.StaticFileHandler ):
        def set_extra_headers(self, path):
            self.set_header( 'Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0' )

    #Define v1 API
    # https://www.tornadoweb.org/en/stable/web.html#tornado.web.RequestHandler.initialize
    translationAPIv1 = [
        # lock, modelDatabase, inputQueue, outputQueue, preloadModel
        (r'/', MainHandler, dict( lock=lock, modelDatabase=modelDatabase, modelDatabaseValidated=modelDatabaseValidated, inputQueue=inputQueue, outputQueue=outputQueue, preloadModel=preloadModel ) ),
        #(r'/', MainHandler),
        (r'/version', ReturnVersion, { 'scriptNameWithVersion' : scriptNameWithVersion } ),
        (r'/api/v2/version', ReturnVersion, { 'scriptNameWithVersion' : scriptNameWithVersion } ),
        (r'/model', ReturnModel, { 'modelDatabase' : modelDatabase, 'modelDatabaseValidated' : modelDatabaseValidated } ),
        (r'/api/v2/model', ReturnModel, { 'modelDatabase' : modelDatabase, 'modelDatabaseValidated' : modelDatabaseValidated } ),
        (r'/api/v2/models', ReturnModelDatabase, { 'modelDatabase' : modelDatabase, 'modelDatabaseValidated' : modelDatabaseValidated } ),
        (r'/api/v2/rawModels', ReturnRawDatabase, { 'modelsTrimmed' : modelsTrimmed } ),
        (r'/api/v2/search', Search, { 'modelDatabase' : modelDatabase, 'modelDatabaseValidated' : modelDatabaseValidated } ),
        #(r'/api/v1/saveCache', SaveCache), # obsolete
        #(r'/api/v1/writeCache', SaveCache), # obsolete
        #(r'/api/v2/clearCache', ClearCache), # useful, but only clears cache for current model
        #(r'/api/v2/getCache', GetCache),       # useful, but only sends cache.zip for current model
        #(r'/resources/ui/(.*)', staticFileHandler, { 'path' : serverSettings[ 'static_path' ] } ),
        #(r'/resources/ui/(.*)', tornado.web.StaticFileHandler, { 'path' : currentScriptPathOnly + '/resources/ui' } ),
        (r'/(.*)', tornado.web.StaticFileHandler, { 'path' : currentScriptPathOnly + '/resources/ui' } ), # If a request is recieved for a static file, then try to serve it from resources/ui as if that was the root. A request for http://server/favicon.ico will search for resources/ui/favicon.ico
        ]

    # Make application that uses the above API. Application can bind to localhost (with IP alias), all addreses, or a specific address.
    # Requiring HostMatches( address ) means that DNS rebind attacks will not work.
    # https://www.tornadoweb.org/en/stable/guide/security.html#dnsrebinding
    if ( address == 'localhost' ) or ( address == '127.0.0.1' ):
        application = tornado.web.Application( [ ( tornado.web.HostMatches( r'(localhost|127\.0\.0\.1)' ), translationAPIv1 ), ], **serverSettings  )
    elif address == '0.0.0.0':
        application = tornado.web.Application( translationAPIv1 )#, **serverSettings )
    else:
        application = tornado.web.Application( [ ( tornado.web.HostMatches( address ), translationAPIv1 ), ], **serverSettings  )

    print( 'Load time: ' + str( round(time.perf_counter() - startedLoadingTime, 2) ) + ' seconds' )
    print()
    print( ( currentScriptNameWithoutPath + ' v' + __version__ ).encode( consoleEncoding ) )
    if preloadModel == True:
        print( ( currentScriptNameWithoutPath + ' ' + modelDatabase[ 'active' ][ 'engine' ] + '.' + modelDatabase[ 'active' ][ 'modelName' ] + ' ' + modelDatabase[ 'active' ][ 'device' ] + ' started: http://' + str( address ) + ':' + str( port ) ).encode( consoleEncoding ) )
    else:
        print( ( currentScriptNameWithoutPath +' started: http://' + str( address ) + ':' + str( port ) ).encode( consoleEncoding ) )
    # if binding to all addresses, then display the connectable addresses for convenience.
    if address == '0.0.0.0':
        print( 'http://localhost:' + str(port) )
        import socket
        if platform.system().lower() == 'windows':
            for i in socket.getaddrinfo( socket.gethostname(), None ):
                #print( 'http://' + str( i[ 4 ][ 0 ] ) + ':' + str( port ) )
                temp = str( i[ 4 ][ 0 ] )
                # filter out IPv6 addresses
                if temp.find( ':' ) == -1:
                    print( 'http://' + temp + ':' + str( port ) )
        else: #Linux
            # On Windows, this prints an error to stderr and stdout returns an array with a single empty string.
            for i in subprocess.run( 'hostname -I', shell=True, capture_output='stdout' ).stdout.decode().strip().split(' '):
                if i.strip() == '':
                    continue
                print( 'http://' + i.strip() + ':' + str( port ) )

    application.listen( address=address, port=port )
    if preloadModel != True:
        # asyncio.get_running_loop().call_later( processTimout, asyncio.get_event_loop().create_task, checkForIdleProcess(), modelDatabase )
        # asyncio.get_event_loop().call_later(10, asyncio.get_event_loop().create_task, checkForIdleProcess( modelDatabase, processTimout ) )
        asyncio.get_event_loop().call_later( callbackResolution, asyncio.get_event_loop().create_task, checkIdleProcess( modelDatabase, lock, unloadTime, inputQueue ) )

#    if cacheEnabled == True:
#        asyncio.get_event_loop().call_later( callbackResolution, asyncio.get_event_loop().create_task, checkIdleCacheWrite( modelDatabase, cacheReadWriteLock, unloadTime, inputQueue ) )

#    try:
    await asyncio.Event().wait()
#    except KeyboardInterrupt:
#        if modelDatabase[ 'active' ] != None:
#            print( 'Stopping NMT process due to KeyboardInterrupt.' )
#            inputQueue.put( 'stopProcess' )
#        #tornado.ioloop.IOLoop.instance().stop()
#        pass

if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        asyncio.run( main() )
    except KeyboardInterrupt:
        tornado.ioloop.IOLoop.instance().stop()
        if psutilAvailable == True:
            for process in psutil.Process( os.getpid() ).children( recursive=True ):
                process.terminate()
        print( 'Program crashed successfully.' )


    sys.exit( 0 )
