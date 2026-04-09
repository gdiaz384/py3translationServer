#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Description: A helper library that has many functions that py3translationServer relies on.

Library Usage: 
import resources.commonFunctions as commonFunctions
#Or to import directly:
import sys
import pathlib
sys.path.append( str( pathlib.Path( 'C:\\resources\\functions.py' ).resolve().parent ) )
import commonFunctions

commonFunctions.verifyThisFileExists()

Function Usage: See each function for usage instructions.

Notes: Only functions that do not use module-wide variables and have return values are listed here. Functions without return values and those that rely on module specific variables that would be cumbersome to pass around should still be in the main program.

Copyright (c) gdiaz384; License: See main program.
"""
__version__ = '2025.06.25'


# Set defaults.
verbose = False
debug = False
consoleEncoding = 'utf-8'
defaultTextFileEncoding = 'utf-8'   # Settings that should not be left as a default setting should have default prepended to them.

inputErrorHandling = 'strict'
#outputErrorHandling = 'namereplace'  # This gets set dynamically below.


import sys                     # outputErrorHandling
import os                       # checkIfThisFileExists(), verifyThisFileExists()
import pathlib               # fixPath()
import configparser     # fixTypesInConfig()
import hashlib              # getSHA1HashOfFile()

try:
    # https://huggingface.co/docs/transformers/main/en/installation#offline-mode
    # https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhubdisabletelemetry
    if os.getenv( 'HF_HUB_OFFLINE' ) != '1':
        os.environ[ 'HF_HUB_OFFLINE' ] = '1'
    if os.getenv( 'HF_HUB_DISABLE_TELEMETRY' ) != '1':
        os.environ[ 'HF_HUB_DISABLE_TELEMETRY' ] = '1'
    import huggingface_hub      #Allows resolving huggingface.co style 'publisher/nmt-model' urls to the huggingface cache on the local filesystem.
    huggingfaceHubAvailable=True
except ImportError:
    huggingfaceHubAvailable=False

#Using the 'namereplace' error handler for text encoding requires Python 3.5+, so use an older one if necessary.
if sys.version_info.minor >= 5:
    outputErrorHandling = 'namereplace'
elif sys.version_info.minor < 5:
    outputErrorHandling = 'backslashreplace'    


# Returns True or False depending upon if myFile, myFolder exists or not.
def checkIfThisFileExists( myFile ):
    if ( myFile == None ) or ( os.path.isfile( str( myFile ) ) != True ):
        return False
    return True

def checkIfThisFolderExists( myFolder ):
    if ( myFolder == None ) or ( os.path.isdir( str( myFolder ) ) != True ):
        return False
    return True

#Usage:
#checkIfThisFileExists( 'myfile.csv' )
#checkIfThisFileExists( myVar )


#Errors out if myFile or myFolder does not exist.
def verifyThisFileExists( myFile, nameOfFileToOutputInCaseOfError=None ):
    if myFile == None:
        print( ( 'Error: Please specify a valid file for: ' + str( nameOfFileToOutputInCaseOfError ) ).encode( consoleEncoding ) )
        sys.exit( 1 )
    if os.path.isfile( myFile ) != True:
        print( ( 'Error: Unable to find file \'' + str( nameOfFileToOutputInCaseOfError ) + '\' ' ).encode( consoleEncoding ) )
        sys.exit( 1 )

def verifyThisFolderExists( myFolder, nameOfFileToOutputInCaseOfError=None ):
    if myFolder == None:
        print( ( 'Error: Please specify a valid folder for: ' + str( nameOfFileToOutputInCaseOfError ) ).encode( consoleEncoding ) )
        sys.exit( 1 )
    if os.path.isdir( myFolder ) != True:
        print( ( 'Error: Unable to find folder \'' + str( nameOfFileToOutputInCaseOfError ) + '\' ' ).encode( consoleEncoding ) )
        sys.exit( 1 )

#Usage:
#verifyThisFileExists( 'myfile.csv', 'myfile.csv' )
#verifyThisFileExists( myVar, 'myVar')
#verifyThisFileExists( myVar )


# This uses pathlib.Path().resolve() to resolve paths that include ~ and relative paths to make them absolute.
# Relative paths become absolute by prepending basePath, the path of the current script, not the os.getcwd().
# Folders with a trailing \ or / will have that character stripped.
def fixPath( path, basePath=str( pathlib.Path( __file__ ).parent.parent ) ):
    if not isinstance( path, str ):
        return None
    if path.find('~') != -1:
        path = str( pathlib.Path( path ).expanduser() )
    if path.startswith( '.\\' ): #On Linux, this is not resolved correctly and becomes basePath + '/.\', so fix it here.
        path='./' + path[ 2 : ].replace( '\\', '/' )
    if pathlib.Path( path ).is_absolute():
        pathIsAbsolute = True
    else:
        if pathlib.Path( path + '/' ).is_absolute(): #for C: => C:/, 'C:'.is_absolute() resolves to False, but 'C:/'.is_absolute() resolves to True #This gives incorrect results for path='/' on Windows
            pathIsAbsolute=True
        else:
            pathIsAbsolute=False
    if ( basePath != None ) and ( pathIsAbsolute == False ):
        path = basePath +'/' + path
    path = str( pathlib.Path( path ).resolve() ) # Resolve gives incorrect results when trying to resolve Windows paths on Linux and vica-versa.

    if ( len( path ) >= 2 ):
        #print( path )
        if ( path[ 1 : 2 ] == ':' ) and ( path[ : 1 ] != '/' ): # heuristic because : can be in folder names on Linux so make sure first character is not /
            assert( ( pathlib.PureWindowsPath( path ).is_absolute() == True ) or ( pathlib.PureWindowsPath( path + '/' ).is_absolute() == True ) )
        # UNC paths that start with \\ are not absolute on Linux.
        elif path.startswith( r'\\' ) == True:
            assert( pathlib.PureWindowsPath( path ).is_absolute() == True )
        else:
            assert( pathlib.PurePosixPath( path ).is_absolute() == True )
    else: #check /
            assert( pathlib.Path( path ).is_absolute() == True )
    return path


def fixTypesInConfig( config ):
    config2={ }
    for section in config:
        #print( '[', section, ']' )
        #if section != 'DEFAULT': #remove this section at the end
        config2[ str(section) ] = { }
        for key in config[ section ]:
            config2[ section ][ key ] = None
            if ( config[ section ][ key ] == '' ) or ( config[ section ][ key ].lower() == 'none' ):
                config2[ section ][ key ] = None
            else:
                try:
                    config2[ section ][ key ]=int( config[ section ][ key ] )
                except:
                    if config[ section ][ key ].lower() in trueList:
                        config2[ section ][ key ] = True
                    elif config[ section ][ key ].lower() in falseList:
                        config2[ section ][ key ] = False
                    else:
                        config2[ section ][ key ] = config[ section ][ key ]
            #config[ section ][ key ]
            #print( 'type(', key, ')', '=', type(config2[ section ][ key ]) )
            #print( key,'=', config2[ section ].get( key+'1' ) ) #This returns None instead of raising a key error.
            #print( key,'=', config2[ section ].get( key ) )
    if 'DEFAULT' in config2:
        config2.pop( 'DEFAULT' )
    #for section in config2:
    #    print( '[', section, ']' )
    #    for key in config2[section]:
    #        print( 'type(', key, ')', '=', type(config2[ section ][ key ]) )
    #        print( key,'=', config2[ section ][ key ] )
    return config2


def getSHA1HashOfFile( filename=None ):
    if filename == None:
        return None
    if not pathlib.Path( filename ).is_file():
        return None
    hash=hashlib.sha1()
    chunk=4194304
    with open( filename , 'rb' ) as myFile:
        for block in iter( lambda: myFile.read( chunk ), b'' ):
            hash.update( block )
    return hash.hexdigest().lower()


def getDevice( device=None ):
    if device != None:
        # transformers uses integers instead of device names, so perform the conversion here to make it ubiquitous.
        try:
            device = int( device )
        except:
            pass
        return device
    try:
        import torch
        if torch.cuda.is_available() == True:
            del torch
            return 'cuda'
        else:
            del torch
            return 'cpu'
    except ImportError:
        return 'cpu'


# This accepts a huggingface model identifier, including as a url, uses huggingface_hub.try_to_load_from_cache() to turn it into a c:\filesystem\path\to\model.
def resolveHuggingfaceUrlToPath( path ):
    if ( huggingfaceHubAvailable != True ) or ( path == None ):
        return path
    resolve = False
    # https://huggingface.co/uploader/model-t
    if ( path.startswith( 'http' ) == True ) and ( path.find( '/' ) != -1 ) and ( path.find( 'huggingface' ) != -1 ):
        resolve = True
    # uploader/model-t
    elif ( path.count( '/' ) == 1 ) and ( path[ : 1 ] != '/' ):
        resolve = True
    if resolve != True:
        return path
    #else:
    if path.count( '/' ) == 1:
        #uploader, model = path.split( '/' )
        returnedPath = huggingface_hub.try_to_load_from_cache( repo_id=path, filename='.gitattributes' )
    else:
        trunk, model = path.rsplit( '/', maxsplit=1 )
        unused, uploader = trunk.rsplit( '/', maxsplit=1 )
        returnedPath = huggingface_hub.try_to_load_from_cache( repo_id=uploader + '/' + model, filename='.gitattributes' )
    if returnedPath != None:
        returnedPath = pathlib.Path( returnedPath )
        if returnedPath.exists():
            return str( returnedPath.parent )
    return path
