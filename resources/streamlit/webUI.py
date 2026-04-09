#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
This is a small web UI for py3translationServer at port=8501 powered by streamlit.

Website: https://streamlit.io 

Usage:
streamlit run webUI.py
streamlit run webUI.py -- -h
streamlit run webUI.py -- --address localhost --port 14366

Notes:
- The address and port are where to send translation requests, not for this UI.
- To change the port of the streamlit UI, use --server.address=  passed to streamlit. Example:
streamlit run myUi.py --server.address=localhost -- --address localhost --port 8080 --quiet

Docs:
https://docs.streamlit.io/get-started/fundamentals/main-concepts
https://forum.opennmt.net/t/simple-web-interface/4527/16
https://github.com/gdiaz384/py3translationServer

Credit: github/gdiaz384 based on the work of opennmt/ymoslem and opennmt/SamuelLacombe.
"""
# Set program defaults.
titleForApp='py3translationServer UI'
tileForPage=titleForApp + '- by gdiaz384'

defaultAddressForTranslation='localhost'
# The webUI runs on port 8501 by default. The port below is where the webapp should send translation requests.
defaultPortForTranslation=14366


#import sys                       # Used to exit in case streamlit is not installed. Import conditionally later.
import argparse                # Used to add command line options.
try:
    import streamlit                 # Main UI
except ImportError:
    print('streamlit did not import successfully. Install with:\n\n pip install streamlit \n')
    import sys
    sys.exit(1)
import requests                 # Send requests to server. streamlist installs requests, so this should always be available.


# Add command line options.
commandLineParser=argparse.ArgumentParser(description='Description: This is a small UI for py3translationServer at port=8501.')
commandLineParser.add_argument('-a', '--address', help='Specify the address of the translation server. Default is: '+ str(defaultAddressForTranslation), default=defaultAddressForTranslation, type=str)
commandLineParser.add_argument('-p', '--port', help='Specify the port of the destination server. Default=' + str(defaultPortForTranslation), default=defaultPortForTranslation, type=int)
commandLineParser.add_argument('-q', '--quiet', help='Silence is golden.', action='store_true')
commandLineParser.add_argument('-d', '--debug', help='Print too much information.', action='store_true')


# Parse command line settings.
commandLineArguments=commandLineParser.parse_args()

address=commandLineArguments.address
port=commandLineArguments.port
quiet=commandLineArguments.quiet
debug=commandLineArguments.debug

hostAddressFull = 'http://' + address + ':' + str(port)
print( 'Translations will be submitted to: ' + hostAddressFull )


def serializeJSON(data=None):
    tempList=[]
    if debug==True:
        print( 'data=' + data )
    # Divive the user input based on new lines in order to submit each line as a different entry.
    if data != None:
        while (data.partition('\n')[0] != None) and (data.partition('\n')[0] != ''):
            tempList.append(data.partition('\n')[0])
            data=data.partition('\n')[2]

    return dict ([ ('content' , tempList ), ('message' , 'translate sentences') ])


streamlit.set_page_config( page_title=tileForPage )
streamlit.title(titleForApp)

#saveCache
#requests.get( hostAddressFull + '/api/v1/saveCache' )  #This runs in an endless loop because streamlit is always reloading the page, so comment it out.

with streamlit.form('saveCacheForm'):
    saveCacheButton = streamlit.form_submit_button("Save Cache")
    if saveCacheButton:
        requests.get( hostAddressFull + '/api/v1/saveCache' )

streamlit.link_button("Download Cache", url=hostAddressFull + '/api/v1/getCache')


# Form to add items.
with streamlit.form('mainTranslationForm'):
    # Add area to type the source text.
    user_input = streamlit.text_area('Source Text')
    submitted = streamlit.form_submit_button("Translate")

    #translation = translate(user_input, translator)
    #translation = requests.post( hostAddressFull, json = translationJSON )
    if ( user_input != None ) and ( user_input != '' ):
        if (__name__ == '__main__' ) and ( quiet != True ):
            print( 'user_input=' + str(user_input) )
        translationJSON = serializeJSON(str(user_input))
        if debug==True:
            print('translationJSON='+translationJSON)

        # Translate.
        translatedList=None
        # Basically, this uses the requests library to send an HTTP POST request to hostAddressFull
        # The data this sends is translationJSON.
        # When it returns, it takes the result and uses the .json() method to get the actual data from the response.body. The result can return either a list or a string depending upon input, but since it was input as a list, the result will always be a list.
        #translatedList = requests.post( hostAddressFull, json = dict([ ('content' , str(user_input) ), ('message' , 'translate sentences') ]) ).json()
        translatedList = requests.post( hostAddressFull, json = translationJSON ).json()
        #translatedList = requests.post( hostAddressFull, json = translationJSON ).text

        if ( submitted ) and ( translatedList != None ):
            if ( __name__ == '__main__' ) and ( quiet != True ):
                print(str(translatedList))
            streamlit.write('Translation Result')
            #streamlit.info(translatedList)

            displayText = ''
            for entry in translatedList:
                displayText=displayText + '\n' + str(entry)
            streamlit.code(displayText, language=None)

            #The following are prettier but they do not respect new lines.
            #streamlit.info(entry)
            #streamlit.write streamlit.code streamlit.success
            #streamlit.code(translatedList)
            #streamlit.success(translatedList)


#Enable word wrap in streamlit's Markdown code blocks.
streamlit.markdown(f""" <style>
   code {{white-space : pre-wrap !important;}}
 </style> """, unsafe_allow_html=True)
