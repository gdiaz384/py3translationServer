#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Description: A library to handle caching output from NMT models. Each instance of MultiLanguageCache is one model and supports multiple language pairs efficently. Saves data to disk in csv.zip format.

import resources.multiLanguageCache
cache = multiLanguageCache.MultiLanguageCache( modelFilenameAndPath='C:/Users/Public/Downloads/nmt/model.bin', modelFriendlyName='nmtModelName', hash=None, cacheFolder=None, csvDialect=None ) # The hash will be calculated dynamically.
cache.clearCache() # Delete all previous entries.
cache.add( untranslatedString, translatedString, sourceLanguage, targetLanguage )
print( cache.get( untranslatedString, sourceLanguage, targetLanguage )
print( cache.cache[ sourceLanguage + '.' + targetLanguage ][ untranslatedString ] )
print( cache.getModelHash() )
print( cache.hash )
cache.save()

Copyright (c) gdiaz384; License: See main program.
"""
__version__ = '2025.06.10'


# This can be a relative path or absolute path.
defaultCacheFolder = './resources/cache'
legacySourceLanguage = 'ja'
legacyTargetLanguage = 'en'
defaultCSVEncoding = 'utf-8'
#None, unix, excel, excel-tab
defaultCSVDialect = None
inputErrorHandling = 'strict'
consoleEncoding = 'utf-8'
#outputErrorHandling = 'namereplace'  # This gets set dynamically below.


import io             # Used to create io.StringIO() used for creating .csv files in memory.
import sys          # Update outputErrorHandling.
import pathlib    # Manipulate paths.
import hashlib   # Calculate hash of model.
import glob        # Search for filenames matching cache.hash.*.csv.
import csv          # i/o cache to disk formatted as .csv files.
import zipfile     # i/o cache to disk as a single compressed file.
import random  # When writing out cache.csv.zip, write it out to a temporary file first.
import datetime # Rename backup files to today's date.

#Using the 'namereplace' error handler for text encoding requires Python 3.5+, so use an older one if necessary.
if sys.version_info.minor >= 5:
    outputErrorHandling = 'namereplace'
elif sys.version_info.minor < 5:
    outputErrorHandling = 'backslashreplace'    


# These functions return the current date, time, yesterday's date, and full (day+time)
def getYearMonthAndDay():
    today = datetime.datetime.today()
    currentYear = str( today.strftime( '%Y' ) )
    currentMonth = str( today.strftime( '%m' ) )
    currentDay = str( today.strftime( '%d' ) )
    return currentYear + '-' + currentMonth + '-' + currentDay #2024-01-24


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
    return hash.hexdigest()


def createDictionaryFromCSVFile( filename, csvEncoding=defaultCSVEncoding, csvDialect=defaultCSVDialect ):
    tempDB = { }
    with open( filename, 'r', newline='', encoding=csvEncoding, errors=inputErrorHandling ) as myFile:
        if csvDialect == None:
            myCsvHandle = csv.reader( myFile )
        else:
            myCsvHandle = csv.reader( myFile, dialect=csvDialect )
            #if csvDialect == 'unix':
            #    myCsvHandle = csv.reader( myFile, dialect='unix' )
            #elif csvDialect == 'excel':
            #    myCsvHandle = csv.reader( myFile, dialect='excel' )
            #elif csvDialect == 'excel-tab':
            #    myCsvHandle = csv.reader( myFile, dialect='excel-tab' )
        for counter,listOfStringsRow in enumerate( myCsvHandle ):
            if counter == 0:
                continue
            for i in range( len( listOfStringsRow ) ):
                # Clean up whitespace for entities. This should not be necessary as long as .strip() is run prior to adding entries to the cache.
                #listOfStringsRow[ i ] = listOfStringsRow[ i ].strip()
                # Fix types.
                if listOfStringsRow[ i ].lower() == 'true':
                    listOfStringsRow[ i ] = True
                elif listOfStringsRow[ i ].lower() == 'false':
                    listOfStringsRow[ i ] = False
                elif ( listOfStringsRow[ i ].lower() == 'none' ) or ( listOfStringsRow[ i ].lower() == '' ):
                    listOfStringsRow[ i ] = None
                # Leave numbers as strings. They should not be processed anyway, so there is no need to mess with them.
            tempDB[ listOfStringsRow[ 0 ] ] = listOfStringsRow[ 1 ]
    return tempDB


def createDictionaryFromListOfStrings( listOfStringsCSVFile, csvDialect=defaultCSVDialect ):
    tempDB = { }
    if csvDialect == None:
        myCsvHandle = csv.reader( listOfStringsCSVFile )
    else:
        myCsvHandle = csv.reader( listOfStringsCSVFile, dialect=csvDialect )
        #if csvDialect == 'unix':
        #    myCsvHandle = csv.reader( listOfStringsCSVFile, dialect='unix' )
        #elif csvDialect == 'excel':
        #    myCsvHandle = csv.reader( listOfStringsCSVFile, dialect='excel' )
        #elif csvDialect == 'excel-tab':
        #    myCsvHandle = csv.reader( listOfStringsCSVFile, dialect='excel-tab' )
    for counter,listOfStringsRow in enumerate( myCsvHandle ):
        if counter == 0:
            continue
        for i in range( len( listOfStringsRow ) ):
            # Clean up whitespace for entities. This should not be necessary as long as .strip() is run prior to adding entries to the cache.
            #listOfStringsRow[ i ] = listOfStringsRow[ i ].strip()
            # Fix types.
            if listOfStringsRow[ i ].lower() == 'true':
                listOfStringsRow[ i ] = True
            elif listOfStringsRow[ i ].lower() == 'false':
                listOfStringsRow[ i ] = False
            elif ( listOfStringsRow[ i ].lower() == 'none' ) or ( listOfStringsRow[ i ].lower() == '' ):
                listOfStringsRow[ i ] = None
            # Leave numbers as strings. They should not be processed anyway, so there is no need to mess with them.
        tempDB[ listOfStringsRow[ 0 ] ] = listOfStringsRow[ 1 ]
    return tempDB


# Cache is based on a filename fed into it at runtime which is the model used to generate data that fills the cache.
# Cache only understands the model used to initalize it.
# Cache can have multiple arbitrary language pairs.
# Within a language pair, cache is source_string:target_string based.
# Filenames are therefore exactly cache.sha1hash.sourceLanguage.targetLanguage.csv
# Multiple .csv files, one for each language pair, get stored in a file called cache.sha1hash.csv.zip
# sourceLanguage and targetLanguage are not drawn from a specific list or validated, but rather whatever is fed to the model at runtime by the user.
class MultiLanguageCache( ):
    # cacheFolder is a folder where the cache should be stored.
    def __init__( self, modelFilenameAndPath=None, modelFriendlyName=None, hash=None, cacheFolder=None, csvDialect=defaultCSVDialect ):
        self.modelpath = modelFilenameAndPath
        assert( pathlib.Path( self.modelpath ).is_file() == True )
        self.modelFriendlyName = modelFriendlyName
        if self.modelFriendlyName == None:
            self.modelFriendlyName = pathlib.Path( modelFilenameAndPath ).name
            # These automatically generated friendly names are not particularly descriptive, so prepend the folder name just in case. If the literal model is stored at the root, c:\pytorch_model.bin, then parent.stem will return an empty string.
            base = str( pathlib.Path( modelFilenameAndPath ).parent.stem )
            if self.modelFriendlyName.lower() == 'model.safetensors':
                self.modelFriendlyName = base + 'model.safetensors'
            if self.modelFriendlyName.lower() == 'pytorch_model.bin':
                self.modelFriendlyName = base + '_pytorch_model.bin'
            elif self.modelFriendlyName.lower() == 'model.bin':
                self.modelFriendlyName = base + '_model.bin'
            elif self.modelFriendlyName.lower() == 'tf_model.h5':
                self.modelFriendlyName = base + '_tf_model.h5'
            elif self.modelFriendlyName.lower() == 'big.pretrain.pt':
                self.modelFriendlyName = base + '_big.pretrain.pt'
            elif self.modelFriendlyName.lower() == 'rust_model.ot':
                self.modelFriendlyName = base + '_rust_model.ot'
            elif self.modelFriendlyName.lower() == 'flax_model.msgpack':
                self.modelFriendlyName = base + '_flax_model.msgpack'
        self.hash = hash
        if self.hash == None:
            self.hash = getSHA1HashOfFile( self.modelpath )
        self.csvDialect = csvDialect
        self.lastSourceLanguage = None
        self.lastTargetLanguage = None

        self.cacheWasUpdated = False
        self.cacheFolder = cacheFolder
        if self.cacheFolder != None:
            self.cacheFolder = cacheFolder
        else: #elif self.cacheFolder == None:
            if pathlib.Path( defaultCacheFolder ).is_absolute() == True:
                self.cacheFolder = str( pathlib.Path( defaultCacheFolder ).resolve() )
            else:
                # Failure case for C:
                if pathlib.Path( defaultCacheFolder + '/' ).is_absolute() == True:
                    # path is actually absolute still
                    self.cacheFolder = str( pathlib.Path( defaultCacheFolder ).resolve() )
                else:
                    #path really is relative
                    self.cacheFolder = str( pathlib.Path( str( pathlib.Path( __file__ ).absolute().parent.parent ) + '/' + defaultCacheFolder ).resolve() )
        #assert( self.cacheFolder != None )
        #print( 'self.cacheFolder=', self.cacheFolder )
        #exit()
        if pathlib.Path( self.cacheFolder ).is_file() == True:
            raise Exception( ( 'Unable to set cacheFolder. Path must be a folder, but a file already exists with that name: ' + self.cacheFolder ).encode( consoleEncoding ) )
        elif pathlib.Path( self.cacheFolder ).exists() == False:
            # if the path does not exist, then create it.
            pathlib.Path( self.cacheFolder ).mkdir( parents=True, exist_ok=True )
        if ( self.cacheFolder[ -1 : ] == '/' ) or ( self.cacheFolder[ -1 : ] == '\\' ):
            self.cacheFolder = self.cacheFolder[ : -1 ]

        #knownFiles is a list of files that match the correct file naming pattern of exactly 'cache.hash.sourceLanguage.targetLanguage.csv' or cache.hash.csv.zip
        knownCSVFiles, knownZipFiles = self._getKnownCSVAndZipFiles( self.cacheFolder, self.hash )
        # Legacy format.
        legacyCSV = self.cacheFolder + '/cache.' + self.hash[ :10 ] + '.csv' 
        if pathlib.Path( legacyCSV ).is_file() == True:
            #print( knownCSVFiles )
            # Rename to proper naming pattern using legacySourceLanguage and legacyTargetLanguage.
            tempTarget = self.cacheFolder + '/cache.' + self.hash + '.' + legacySourceLanguage + '.' + legacyTargetLanguage + '.csv'
            print( ( 'renaming ' + legacyCSV + ' to ' + tempTarget ).encode( consoleEncoding ) )
            pathlib.Path( legacyCSV ).replace( tempTarget )
            #append to self.knownFiles
            knownCSVFiles.append( ( pathlib.Path( tempTarget ).name, 'csv', legacySourceLanguage, legacyTargetLanguage ) )

        self.cache = { }
        #print( knownCSVFiles )
        for entryList in knownCSVFiles:
            #entryList = ( filename, type (csv or csv.zip), sourceLanguage, targetLanguage )
            tempFilename = entryList[ 0 ]
            assert( entryList[ 1 ] == 'csv' )
            #print( entryList )
            #print( tempFilename )
            #createDictionaryFromCSVFile( filename, csvEncoding=defaultCSVEncoding, csvDialect=defaultCSVDialect ):
            tempDB = createDictionaryFromCSVFile( self.cacheFolder + '/' + tempFilename, csvEncoding=defaultCSVEncoding, csvDialect=self.csvDialect )
            tempSourceLanguageCode = entryList[ 2 ].lower()
            tempTargetLanguageCode = entryList[ 3 ].lower()
            tempIndex = tempSourceLanguageCode + '.' + tempTargetLanguageCode
            if tempIndex in self.cache:
                self.cache[ tempIndex ].update( tempDB) #tempDB will overwrite whatever is in self.cache[ tempIndex ]
            else:
                self.cache[ tempIndex ] = tempDB

        for entryList in knownZipFiles:
            #entryList = ( filename, type (csv or csv.zip) )
            tempFilename = entryList[ 0 ]
            assert( entryList[ 1 ] == 'csv.zip' )
            tempCSVFiles = [ ]
            with zipfile.ZipFile( self.cacheFolder + '/' + tempFilename, mode='r' ) as myzip:
                print( ( 'List of files in ' + str( tempFilename ) + ':' + str( myzip.namelist( ) ) ).encode( consoleEncoding ) )
                for entryName in myzip.namelist():
                    # cache.hash.sourceLanguage.targetLanguage.csv
                    entryNameList = entryName.split( '.' )
                    if entryNameList[ 0 ].lower() != 'cache':
                        continue
                    if len( entryNameList ) != 5:
                        continue
                    if entryNameList[ 4 ].lower() != 'csv':
                        continue

                    # [filename.csv, sourceLanguage, targetLanguage ]
                    metadata = [ entryName, entryNameList[ 2 ], entryNameList[ 3 ] ]
                    tempfileContents = myzip.read( entryName ).decode( defaultCSVEncoding )

                    # Determine newline encoding.
                    index = tempfileContents.find('\n')
                    if index == -1:
                        newline = '\r' #old mac
                    elif tempfileContents[ index - 1 : index ] == '\r':
                        newline = '\r\n' #windows
                    else:
                        newline = '\n' #linux
                    tempfileContents=tempfileContents.split( newline )
                    # Remove last empty line.
                    if len( tempfileContents[ -1 : ][ 0 ].strip() ) == 0:
                        tempfileContents = tempfileContents[ : -1 ]

                    tempCSVFiles.append( ( metadata, tempfileContents ) )

            for csvFileAsListOfStrings in tempCSVFiles:
                #createDictionaryFromListOfStrings( listOfStringsCSVFile, csvDialect=defaultCSVDialect ):
                tempDB = createDictionaryFromListOfStrings( csvFileAsListOfStrings[1], csvDialect=self.csvDialect )
                tempSourceLanguageCode = csvFileAsListOfStrings[ 0 ][ 1 ].lower()
                tempTargetLanguageCode = csvFileAsListOfStrings[ 0 ][ 2 ].lower()
                tempIndex = tempSourceLanguageCode + '.' + tempTargetLanguageCode
                if tempIndex in self.cache:
                    #Syntax: self.cache[ tempIndex ].update( tempDB) # tempDB will overwrite whatever is in self.cache[ tempIndex ]
                    #tempDB.update( self.cache[ tempIndex ] ) # self.cache[ tempIndex ] will overwrite whatever is in tempDB
                    #self.cache[ tempIndex ] = tempDB # Add only missing entries to allow for partial updates.
                    pass #Do not add missing entries so that anything deleted by the user stays deleted.
                else:
                    self.cache[ tempIndex ] = tempDB

    # Zero out cache.
    # Returns True if cache was cleared, or False if it was not cleared.
    def clearCache( self ):
        self.cache = { }
        return True

    # Update cache. Access as mlc.add()
    def add( self, untranslatedString, translatedString, sourceLanguage=None, targetLanguage=None, overwrite=False ):
        if ( isinstance( untranslatedString, str ) == False ) or ( isinstance( translatedString, str ) == False ):
            return False
        untranslatedString = untranslatedString.strip()
        translatedString = translatedString.strip()
        if ( untranslatedString == '' ) or ( translatedString == '' ):
            return False

        if ( self.lastSourceLanguage == None ) and ( sourceLanguage == None ):
            return False
        elif ( self.lastSourceLanguage != None ) and ( sourceLanguage != None ):
            if self.lastSourceLanguage != sourceLanguage:
                self.lastSourceLanguage = sourceLanguage
        elif ( self.lastSourceLanguage == None ) and ( sourceLanguage != None ):
            self.lastSourceLanguage = sourceLanguage
        elif ( self.lastSourceLanguage != None ) and ( sourceLanguage == None ):
            sourceLanguage = self.lastSourceLanguage

        if ( self.lastTargetLanguage == None ) and ( targetLanguage == None ):
            return False
        elif ( self.lastTargetLanguage != None ) and ( targetLanguage != None ):
            if self.lastTargetLanguage != targetLanguage:
                self.lastTargetLanguage = targetLanguage
        elif ( self.lastTargetLanguage == None ) and ( targetLanguage != None ):
            self.lastTargetLanguage = targetLanguage
        elif ( self.lastTargetLanguage != None ) and ( targetLanguage == None ):
            targetLanguage = self.lastTargetLanguage

        languagePair = sourceLanguage + '.' + targetLanguage
        if languagePair in self.cache == False:
            self.cache[ languagePair ] = { }

        if ( untranslatedString in self.cache[ languagePair ] ) == True:
            if overwrite == True:
                self.cache[ languagePair ][ untranslatedString ] = translatedString
            return True
        #else:
        self.cache[ languagePair ][ untranslatedString ] = translatedString
        if self.cacheWasUpdated != True:
            self.cacheWasUpdated = True
        return True

    # Retrieve entries from cache. Access as mlc.get()
    # It is also possible to get the same information using cache.cache[ languagePair ][ untranslatedString ]
    def get( self, untranslatedString, sourceLanguage, targetLanguage ):
        languagePair = sourceLanguage + '.' + targetLanguage
        if languagePair in self.cache:
            if untranslatedString in self.cache[ languagePair ]:
                return self.cache[ languagePair ][ untranslatedString ]
        return None

    # Write out to filesystem.
    def save( self ):
        if len( self.cache ) == 0:
            return False
        # Get known csv and csv.zip files.
        knownCSVFiles, knownZipFiles = self._getKnownCSVAndZipFiles( self.cacheFolder, self.hash )
        # Write out to temporary file.
        outputZipFilename = 'cache.' + self.hash + '.csv.zip'
        outputZipFilenameTemp = 'cache.' + self.hash + '.' + str( random.randint( 10000, 99999 ) ) + '.csv.zip'
        with zipfile.ZipFile( self.cacheFolder + '/' + outputZipFilenameTemp, mode='w', compression=zipfile.ZIP_LZMA ) as myzip:
            for key,value in self.cache.items():
                # key is a language pair, as in self.cache[ languagePair ]
                # value is the DB { } itself that has the mappings
                if len( self.cache[ key ] ) == 0:
                    continue
                tempLanguagePair = key.split( '.' )
                assert( len( tempLanguagePair ) == 2 )
                tempSourceLanguageCode = tempLanguagePair[ 0 ]
                tempTargetLanguageCode = tempLanguagePair[ 1 ]
                tempFilename = 'cache.' + self.hash + '.' + tempSourceLanguageCode + '.' + tempTargetLanguageCode + '.csv'
                tempHeader = [ 'rawdata', self.modelFriendlyName + '#' + self.hash ]
                # Convert cache DB { } back to list of strings.
                tempData = [ ]
                for key2,value2 in value.items():
                    tempData.append( [ key2, value2 ] )
                # Create in-memory text file.
                tempFile = io.StringIO()
                if self.csvDialect == None:
                    csv.writer( tempFile ).writerow( tempHeader )
                    csv.writer( tempFile ).writerows( tempData )
                else:
                    csv.writer( tempFile, dialect=self.csvDialect ).writerow( tempHeader )
                    csv.writer( tempFile, dialect=self.csvDialect ).writerows( tempData )
                # Write memory.csv file to .zip.
                myzip.writestr( tempFilename, tempFile.getvalue() )
                tempFile.close()
        print( ( 'Wrote: ' + self.cacheFolder + '/' + outputZipFilenameTemp ).encode( consoleEncoding ) )
        if pathlib.Path( outputZipFilenameTemp ).exists() == True:
            self.cacheWasUpdated = False

        # Rename any existing csv files to csv.backup.d-at-e.csv. Do not rename csv.zip to csv.zip.backup.d-at-e.zip unless that file is getting written out.
        for entryList in knownCSVFiles:
            #( filename, type (csv or csv.zip), sourceLanguage, targetLanguage )
            tempFilename = entryList[ 0 ]
            tempFilenameTarget = tempFilename + '.backup.' + getYearMonthAndDay() + '.csv'
            # Rename/replace temporary file to target destination.
            pathlib.Path( self.cacheFolder + '/' + tempFilename ).replace( self.cacheFolder + '/' + tempFilenameTarget )
            # Print out message to console that it was renamed.
            print( ( 'renamed ' + tempFilename + 'to' + tempFilenameTarget ).encode( consoleEncoding ) )

        if ( pathlib.Path( self.cacheFolder + '/' + outputZipFilename ).exists() == True ):
            outputZipFilenameTemp2 = outputZipFilename + '.backup.' + getYearMonthAndDay() + '.csv.zip'
            pathlib.Path( self.cacheFolder + '/' + outputZipFilename ).replace( self.cacheFolder + '/' + outputZipFilenameTemp2 )
            print( ( 'Renamed ' + outputZipFilename + ' to ' + outputZipFilenameTemp2 ).encode( consoleEncoding ) )
        pathlib.Path( self.cacheFolder + '/' + outputZipFilenameTemp ).replace( self.cacheFolder + '/' + outputZipFilename )

    def _getKnownCSVAndZipFiles( self, cacheFolder, hash ):
        fileList = glob.glob( cacheFolder + '/*' )
        validatedCSVList = [ ]
        validatedZipList = [ ]
        for entry in fileList:
            if pathlib.Path( entry ).is_file() == False:
                continue
            entry = pathlib.Path( entry ).name
            # cache.sha1hash.sourceLanguage.targetLanguage.csv
            # cache.sha1hash.csv.zip
            tempList = entry.split( '.' )
            if tempList[ 0 ].lower() != 'cache':
                #print(entry)
                continue
            if ( len( tempList ) != 5 ) and ( len( tempList ) != 4 ) :
                #print(entry)
                continue
            if tempList[ 1 ] != hash:
                continue
            if len( tempList ) == 5:
                if tempList[ 4 ].lower() != 'csv':
                    continue
            if len( tempList ) == 4:
                if ( tempList[ 2 ].lower() != 'csv' ) or ( tempList[ 3 ].lower() != 'zip' ):
                    continue
            # filename, sourceLanguage, targetLanguage, type (csv)
            # filename, type (csv.zip)
            if len( tempList ) == 5:
                validatedCSVList.append( ( entry, 'csv', tempList[ 2 ], tempList[ 3 ] ) )
            #elif len( tempList ) == 4:
            else:
                validatedZipList.append( ( entry, 'csv.zip' ) )
        #print(validatedCSVList)
        #print()
        #print(validatedZipList)
        return validatedCSVList, validatedZipList

