"""
parseReadmeToCSV.py converts some of the data in the readme.md files for models from huggingface.co/Helsinki-NLP to .csv format.

Example for https://huggingface.co/Helsinki-NLP/opus-mt-zh-en
modelName: opus-mt-zh-en
modelPath: Helsinki-NLP/opus-mt-zh-en
sourceLanguages: cjy_Hans cjy_Hant cmn cmn_Hans cmn_Hant gan lzh lzh_Hans nan wuu yue yue_Hans yue_Hant
targetLanguages: eng
modelURL: https://huggingface.co/Helsinki-NLP/opus-mt-zh-en
description: This model can be used for translation and text-to-text generation.
prependTargetLanguage: False

Parsing is done one category of models at at time. parseReadmeToCSV.py requires various preformatted lists, one for each category. These required lists are generated from copying data from fetchReadme.py's opusmtMasterList.txt into the following files, each representing a different category.
opusmttcbiblebigFilename = 'lists-2025-06-26/opus-mt-tc-bible-big.txt'
opusmttcbigFilename = 'lists-2025-06-26/opus-mt-tc-big.txt'
opusmttcbaseFilename = 'lists-2025-06-26/opus-mt-tc-base.txt'
opustatoebaFilename = 'lists-2025-06-26/opus-tatoeba.txt'
opusmtFilename = 'lists-2025-06-26/opus-mt.txt'

Docs:
https://huggingface.co/docs/huggingface_hub/en/guides/download
https://huggingface.co/docs/huggingface_hub/main/en/guides/search
https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_api#huggingface_hub.HfApi.list_models
https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py#L2033
https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_files
https://huggingface.co/docs/huggingface_hub/main/en/package_reference/cache#huggingface_hub.try_to_load_from_cache
https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models
https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache
"""
__version__ = '2025.06.26'


import sys
import pathlib
import csv

if os.getenv('HF_HUB_DISABLE_TELEMETRY') != '1':
    os.environ[ 'HF_HUB_DISABLE_TELEMETRY' ] = '1'
import huggingface_hub


readmeFilename = 'README.md'
huggingfacePrefix='https://huggingface.co/'
opusmttcbiblebigFilename = 'lists-2025-06-26/opus-mt-tc-bible-big.txt'
opusmttcbigFilename = 'lists-2025-06-26/opus-mt-tc-big.txt'
opusmttcbaseFilename = 'lists-2025-06-26/opus-mt-tc-base.txt'
opustatoebaFilename = 'lists-2025-06-26/opus-tatoeba.txt'
opusmtFilename = 'lists-2025-06-26/opus-mt.txt'
otherFilename = 'lists-2025-06-26/other.txt'
debug = False


def readFile( filename ):
    with open(filename, 'rt', encoding='utf-8' ) as file:
        readmeString = file.read()
    return readmeString


def writeCSV( filename, tempList ):
    assert( isinstance( tempList[ 0 ], list ) == True )
    if pathlib.Path( filename ).is_file() == True:
        return
    header=[ 'modelName', 'modelPath', 'sourceLanguages', 'targetLanguages', 'modelURL', 'description', 'prependTargetLanguage' ]
    with open( filename, 'w', newline='', encoding='utf-8' ) as outfile:
        myCsvHandle = csv.writer( outfile )
        myCsvHandle.writerow( header )
        for row in tempList:
            myCsvHandle.writerow( row )
    print( 'Wrote: ', filename )


# This returns the characters between startHere and endHere, excluding the search strings.
def getStringFromFile( file, startHere, endHere ):
    startIndex = file.find( startHere )
    if startIndex == -1:
        return None
    endIndex = file.find( endHere )
    if endIndex == -1:
        return None
    string=file[ startIndex + len( startHere ) : endIndex ].strip()
    if string == '':
        return None
    return string


# Opus MT TC Bible Big #
def opusmttcbiblebigParser( readmeString, repositoryID ):
    description = getStringFromFile( readmeString, '## Model Details', 'This model is part of the [OPUS-MT project]' )
    sourceLanguages = getStringFromFile( readmeString, 'Source Language(s):', '- Target Language(s):' )
    targetLanguages=getStringFromFile( readmeString, '- Target Language(s): ', '- **Original Model**:' )
    if targetLanguages.find( '\n' ) != -1:
        targetLanguages = targetLanguages.partition( '\n' )[ 0 ].strip()

    modelName = repositoryID.split( '/' )[ 1 ]
    url = huggingfacePrefix + repositoryID
    if len( targetLanguages.split(' ') ) == 1:
        prependTargetLanguage = False
    else:
        prependTargetLanguage = True

    tempList = [ modelName, repositoryID, sourceLanguages, targetLanguages, url, description, prependTargetLanguage ]
    return tempList

    debug = False
    if debug == True:
        print( 'description:', description )
        print()
        print( 'sourceLanguages:', sourceLanguages )
        print()
        print( 'targetLanguages:', targetLanguages)
        print()


if pathlib.Path( opusmttcbiblebigFilename + '.csv' ).is_file() == False:
    opusmttcbiblebigList = readFile( opusmttcbiblebigFilename ).strip().splitlines()
    opusmttcbiblebigList = sorted( opusmttcbiblebigList )
    opusmttcbiblebigParsedData = [ ] # Should be [ [ ], [ ] ] where every inner list represents data parsed from 1 file.

    for repositoryID in opusmttcbiblebigList:
        print( 'Parsing:', repositoryID )
        filename = huggingface_hub.try_to_load_from_cache( repo_id=repositoryID, filename=readmeFilename )
        repositoryReadme = readFile( filename )
        opusmttcbiblebigParsedData.append( opusmttcbiblebigParser( repositoryReadme, repositoryID ) )
    #    for i in opusmttcbiblebigParsedData[ 0 ]:
    #        print( i )
        #break

    print( opusmttcbiblebigParsedData )
    writeCSV( opusmttcbiblebigFilename + '.csv' , opusmttcbiblebigParsedData )


# Opus MT TC Big #
def opusmttcbigParser( readmeString, repositoryID ):
    # Strange parsing for opus-mt-tc-big-gmq-gmq.
    modelName = repositoryID.split( '/' )[ 1 ]
    url = huggingfacePrefix + repositoryID

    description = getStringFromFile( readmeString, '# ' + modelName.lower(), 'This model is part of the [OPUS-MT project]' )
    if description == None:
        description = getStringFromFile( readmeString, '## Model Details', 'This model is part of the [OPUS-MT project]' )
    sourceLanguages = getStringFromFile( readmeString, '* source language(s): ', '* target language(s):' )
    if sourceLanguages == None:
        sourceLanguages = getStringFromFile( readmeString, '- Source Language(s): ', ' - Target Language(s): ' )
    targetLanguages = getStringFromFile( readmeString, '* target language(s): ', '* model: ' )
    if targetLanguages == None:
        targetLanguages = getStringFromFile( readmeString, ' - Target Language(s): ', ' - Language Pair(s): ' )

    # In the following models, the sourceLanguages and targetLanguages are blank in the transformers readme, so just hardcode fixes for them based on their Tatoeba-Challenge readme.md.
    # https://github.com/Helsinki-NLP/Tatoeba-Challenge/blob/master/models/eng-kor/README.md
    if repositoryID == 'Helsinki-NLP/opus-mt-tc-big-en-ko':
        sourceLanguages = 'eng'
        targetLanguages = 'kor'
    # https://github.com/Helsinki-NLP/Tatoeba-Challenge/blob/master/models/kor-eng/README.md
    elif repositoryID == 'Helsinki-NLP/opus-mt-tc-big-ko-en':
        sourceLanguages = 'kor'
        targetLanguages = 'eng'

    if description.find( '\n' ) != -1:
        description=description.rsplit( '\n', maxsplit=1 )[ 1 ].strip()
    if targetLanguages.find( '\n' ) != -1:
        targetLanguages = targetLanguages.partition( '\n' )[ 0 ].strip()

    if description == None:
        print( 'Error processing description for', repositoryID )
        sys.exit( 1 )
    if sourceLanguages == None:
        print( 'Error processing sourceLanguages for', repositoryID )
        sys.exit( 1 )
    if targetLanguages == None:
        print( 'Error processing targetLanguages for', repositoryID )
        sys.exit( 1 )

    if len( targetLanguages.split(' ') ) == 1:
        prependTargetLanguage = False
    else:
        prependTargetLanguage = True

    tempList = [ modelName, repositoryID, sourceLanguages, targetLanguages, url, description, prependTargetLanguage ]
    return tempList


if pathlib.Path( opusmttcbigFilename + '.csv' ).is_file() == False:
    opusmttcbigList = readFile( opusmttcbigFilename ).strip().splitlines()
    opusmttcbigList = sorted( opusmttcbigList )
    opusmttcbigParsedData = [ ] # Should be [ [ ], [ ] ] where every inner list represents data parsed from 1 file.

    for repositoryID in opusmttcbigList:
        print( 'Parsing:', repositoryID )
        filename = huggingface_hub.try_to_load_from_cache( repo_id=repositoryID, filename=readmeFilename )
        repositoryReadme = readFile( filename )
        opusmttcbigParsedData.append( opusmttcbigParser( repositoryReadme, repositoryID ) )
        #for i in opusmttcbigParsedData:
        #    print( i )
        #break

    #for i in opusmttcbigParsedData:
    #    print( i )
    writeCSV( opusmttcbigFilename + '.csv' , opusmttcbigParsedData )


# Opus MT TC Base #
def opusmttcbaseParser( readmeString, repositoryID ):
    # Strange parsing for opus-mt-tc-big-gmq-gmq.
    modelName = repositoryID.split( '/' )[ 1 ]
    url = huggingfacePrefix + repositoryID

    description = getStringFromFile( readmeString, '# ' + modelName.lower(), 'This model is part of the [OPUS-MT project]' )
    sourceLanguages = getStringFromFile( readmeString, '* source language(s): ', '* target language(s):' )
    if sourceLanguages == None:
        sourceLanguages = getStringFromFile( readmeString, '- Source Language(s): ', ' - Target Language(s): ' )
    targetLanguages = getStringFromFile( readmeString, '* target language(s): ', '* model: ' )
    if targetLanguages == None:
        targetLanguages = getStringFromFile( readmeString, ' - Target Language(s): ', ' - Language Pair(s): ' )

    # In the following models, the sourceLanguages and/or targetLanguages are blank in the transformers readme, so just hardcode fixes for them based on their Tatoeba-Challenge readme.md.
    # https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/ukr-ron/README.md
    if repositoryID == 'Helsinki-NLP/opus-mt-tc-base-ro-uk':
        sourceLanguages = 'ukr'
        targetLanguages = 'mol ron'
    # https://github.com/Helsinki-NLP/Tatoeba-Challenge/blob/master/models/tur-ukr/README.md
    elif repositoryID == 'Helsinki-NLP/opus-mt-tc-base-tr-uk':
        sourceLanguages = 'tur'
    # https://github.com/Helsinki-NLP/Tatoeba-Challenge/blob/master/models/ukr-ron/README.md
    elif repositoryID == 'Helsinki-NLP/opus-mt-tc-base-uk-ro':
        sourceLanguages = 'ukr'
        targetLanguages = 'mol ron'
    # https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/ukr-tur/README.md
    elif repositoryID == 'Helsinki-NLP/opus-mt-tc-base-uk-tr':
        targetLanguages = 'tur'
    # https://github.com/Helsinki-NLP/Tatoeba-Challenge/blob/master/models/ces%2Bslk-ukr/README.md
    elif repositoryID == 'Helsinki-NLP/opus-mt-tc-base-ces_slk-uk':
        # This one is missing 'slk' in the sourceLanguages which is present in the other Tatoeba readme.
        sourceLanguages = sourceLanguages + ' slk'
    # https://github.com/Helsinki-NLP/Tatoeba-Challenge/blob/master/models/ukr-ces%2Bslk/README.md
    elif repositoryID == 'Helsinki-NLP/opus-mt-tc-base-uk-ces_slk':
        # This one is missing 'slk' in the targetLanguages which is present in the other Tatoeba readme.
        targetLanguages = 'slk ' + targetLanguages

    if description.find( '\n' ) != -1:
        description=description.rsplit( '\n', maxsplit=1 )[ 1 ].strip()
    if targetLanguages.find( '\n' ) != -1:
        targetLanguages = targetLanguages.split( '\n' )[ 0 ].strip()

    if description == None:
        print( 'Error processing description for', repositoryID )
        sys.exit( 1 )
    if sourceLanguages == None:
        print( 'Error processing sourceLanguages for', repositoryID )
        sys.exit( 1 )
    if ( targetLanguages == None ) or ( targetLanguages == '* valid target language labels:' ):
        print( 'Error processing targetLanguages for', repositoryID )
        sys.exit( 1 )

    if len( targetLanguages.split(' ') ) == 1:
        prependTargetLanguage = False
    else:
        prependTargetLanguage = True

    tempList = [ modelName, repositoryID, sourceLanguages, targetLanguages, url, description, prependTargetLanguage ]
    return tempList


if pathlib.Path( opusmttcbaseFilename + '.csv' ).is_file() == False:
    opusmttcbaseList = readFile( opusmttcbaseFilename ).strip().splitlines()
    opusmttcbaseList = sorted( opusmttcbaseList )
    opusmttcbaseParsedData = [ ] # Should be [ [ ], [ ] ] where every inner list represents data parsed from 1 file.

    for repositoryID in opusmttcbaseList:
        print( 'Parsing:', repositoryID )
        filename = huggingface_hub.try_to_load_from_cache( repo_id=repositoryID, filename=readmeFilename )
        repositoryReadme = readFile( filename )
        opusmttcbaseParsedData.append( opusmttcbaseParser( repositoryReadme, repositoryID ) )
        #for i in opusmttcbaseParsedData:
        #    print( i )
        #break

    #for i in opusmttcbaseParsedData:
    #    print( i )
    writeCSV( opusmttcbaseFilename + '.csv' , opusmttcbaseParsedData )


# Opus Totoeba #
def opustotoebaParser( readmeString, repositoryID ):
    modelName = repositoryID.split( '/' )[ 1 ]
    url = huggingfacePrefix + repositoryID

    description = getStringFromFile( readmeString, '---\n### ', '*  OPUS readme:' )
    sourceLanguages = getStringFromFile( readmeString, '* source language(s): ', '* target language(s):' )
    if sourceLanguages == None:
        sourceLanguages = getStringFromFile( readmeString, '- source_languages: ', '- target_languages: ' )
    targetLanguages = getStringFromFile( readmeString, '* target language(s):', '* model:' )
    if targetLanguages == None:
        targetLanguages = getStringFromFile( readmeString, '- target_languages: ', '- opus_readme_url:' )

    if targetLanguages.find( '\n' ) != -1:
        targetLanguages = targetLanguages.split( '\n' )[ 0 ].strip()

    if description == None:
        print( 'Error processing description for', repositoryID )
        sys.exit( 1 )
    if sourceLanguages == None:
        print( 'Error processing sourceLanguages for', repositoryID )
        sys.exit( 1 )
    if ( targetLanguages == None ):  # or ( targetLanguages == '* valid target language labels:' ):
        print( 'Error processing targetLanguages for', repositoryID )
        sys.exit( 1 )

    description = description.replace( '\n', ' ' ).replace( '  ', ' ' ).replace( ' * ', ', ' ) + ' NMT model.'
    if len( targetLanguages.split(' ') ) == 1:
        prependTargetLanguage = False
    else:
        prependTargetLanguage = True

    tempList = [ modelName, repositoryID, sourceLanguages, targetLanguages, url, description, prependTargetLanguage ]
    return tempList


if pathlib.Path( opustatoebaFilename + '.csv' ).is_file() == False:
    opustotoebaList = readFile( opustatoebaFilename ).strip().splitlines()
    opustotoebaList = sorted( opustotoebaList )
    opustotoebaParsedData = [ ] # Should be [ [ ], [ ] ] where every inner list represents data parsed from 1 file.

    for repositoryID in opustotoebaList:
        print( 'Parsing:', repositoryID )
        filename = huggingface_hub.try_to_load_from_cache( repo_id=repositoryID, filename=readmeFilename )
        repositoryReadme = readFile( filename )
        opustotoebaParsedData.append( opustotoebaParser( repositoryReadme, repositoryID ) )
        #for i in opustotoebaParsedData:
        #    print( i )
        #break

    #for i in opustotoebaParsedData:
    #    print( i )
    writeCSV( opustatoebaFilename + '.csv' , opustotoebaParsedData )


# Opus MT #
def opusmtParser( readmeString, repositoryID ):
    modelName = repositoryID.split( '/' )[ 1 ]
    url = huggingfacePrefix + repositoryID

    if readmeString.find( '* source group:' ) != -1:
        description = getStringFromFile( readmeString, '###', '*  OPUS readme:' )
        if description != None:
            description = description.replace( '\n', ' ' ).replace( '  ', ' ' ).replace( ' * ', ', ' ) + ' NMT model.'
    else:
        description = getStringFromFile( readmeString, '###', '* source languages:' )
        if description != None:
            description = description + ' NMT model.'
    if description == None:
        description = getStringFromFile( readmeString, '#### Direct Use', '## Risks, Limitations and Biases' )
    if description == None:
        description = getStringFromFile(readmeString, '###', '* OPUS readme:' )
        if description != None:
            description = description.replace( '\n', ' ' ).replace( '  ', ' ' ).replace( ' * ', ', ' ) + ' NMT model.'
    
    sourceLanguages = getStringFromFile( readmeString, '* source language(s):', '* target language(s):' )
    if sourceLanguages == None:
        sourceLanguages = getStringFromFile( readmeString, '* source languages:', '* target languages:' )
    if sourceLanguages == None:
        sourceLanguages = getStringFromFile( readmeString, '- Source Language:', '- Target Language:' )
    if sourceLanguages == None:
        sourceLanguages = getStringFromFile( readmeString, '* source language codes:', '* target language codes:' )

    targetLanguages = getStringFromFile( readmeString, '* target language(s):', 'pre-processing:' )
    if targetLanguages == None:
        targetLanguages = getStringFromFile( readmeString, '* target languages:', 'OPUS readme:' )
    if targetLanguages == None:
        targetLanguages = getStringFromFile( readmeString, '- Target Language:', '- **License:**' )
    if targetLanguages == None:
        targetLanguages = getStringFromFile( readmeString, '* target language codes:', '* dataset' )

    opusReadmeLink=None
    # In the following models, some of the info is missing in the transformers readme, so just hardcode fixes for them based on their Tatoeba-Challenge readme.md.
    # https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/heb-fra
    if repositoryID == 'Helsinki-NLP/opus-mt-he-fr':
        sourceLanguages = 'heb'
        targetLanguages = 'fra'
        description='OPUS dataset, source language(s): heb, target language(s): fra NMT model.'
        opusReadmeLink='https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/heb-fra'
    # https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/zho-eng/README.md
    elif repositoryID == 'Helsinki-NLP/opus-mt-zh-en':
        sourceLanguages = 'cjy_Hans cjy_Hant cmn cmn_Hans cmn_Hant gan lzh lzh_Hans nan wuu yue yue_Hans yue_Hant'
        targetLanguages = 'eng'

    if opusReadmeLink == None:
        #opusReadmeLink = getStringFromFile( readmeString, 'OPUS readme:', 'model:' )
        opusReadmeLink = readmeString.partition( 'OPUS readme:' )[ 2 ]
        opusReadmeLink = getStringFromFile( opusReadmeLink, '(', ')' )
        if ( opusReadmeLink.startswith( 'https://github.com/' ) == False ) and ( opusReadmeLink.startswith( 'https://object.pouta.csc.fi/Tatoeba-MT-models/' ) == False ):
            print( 'Error processing opusReadmeLink for', repositoryID )
            print( opusReadmeLink )
            sys.exit( 1 )
    url = url + ', ' + opusReadmeLink

    if description == None:
        print( 'Error processing description for', repositoryID )
        sys.exit( 1 )
    if sourceLanguages == None:
        print( 'Error processing sourceLanguages for', repositoryID )
        sys.exit( 1 )
    if ( targetLanguages == None ):  # or ( targetLanguages == '* valid target language labels:' ):
        print( 'Error processing targetLanguages for', repositoryID )
        sys.exit( 1 )

    if description.find( '\n' ) != -1:
        description = description.replace( '\n', ' ' ).replace( '  ', ' ' ).replace( ' * ', ', ' )# + ' NMT model.'
    if sourceLanguages.find( ',' ) != -1:
        sourceLanguages = sourceLanguages.replace( ',', ' ' )
    if targetLanguages.find( '\n' ) != -1:
        targetLanguages = targetLanguages.split( '\n' )[ 0 ].strip()
    if targetLanguages.find( ',' ) != -1:
        targetLanguages = targetLanguages.replace( ',', ' ' )

    if len( targetLanguages.split(' ') ) == 1:
        prependTargetLanguage = False
    else:
        prependTargetLanguage = True

    tempList = [ modelName, repositoryID, sourceLanguages, targetLanguages, url, description, prependTargetLanguage ]
    return tempList


if pathlib.Path( opusmtFilename + '.csv' ).is_file() == False:
    opusmtList = readFile( opusmtFilename ).strip().splitlines()
    opusmtList = sorted( opusmtList )
    opusmtParsedData = [ ] # Should be [ [ ], [ ] ] where every inner list represents data parsed from 1 file.

    for repositoryID in opusmtList:
        print( 'Parsing:', repositoryID )
        filename = huggingface_hub.try_to_load_from_cache( repo_id=repositoryID, filename=readmeFilename )
        repositoryReadme = readFile( filename )
        opusmtParsedData.append( opusmtParser( repositoryReadme, repositoryID ) )
        #for i in opusmtParsedData:
        #    print( i )
        #break

    for i in opusmtParsedData:
        print( i )
    writeCSV( opusmtFilename + '.csv' , opusmtParsedData )

