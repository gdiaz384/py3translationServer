@echo off
setlocal enabledelayedexpansion
::
:: This script attempts to convert fairseq models (big.pretrain.pt) to the format required by CTranslate2 (model.bin).
::
:: ct2-fairseq-converter.exe must be installed. It can be installed after installing Python 3 by using pip. Syntax:
:: pip install ctranslate2 scikit-learn
:: Entering the above command into a command prompt will install ct2-fairseq-converter.exe to the Scripts\ folder of the Python 3 environment.
::
:: Usage: 0) fairseqToCTranslate2_converter.bat must be run inside of a folder that has python.exe available and where that resulting Python
:: environment has access to the PyTorch library. In other words, copy this file to the appropriate location for portable versions of Python,
:: or install PyTorch globally. Instructions: https://pytorch.org   Select pip + Python.
:: 1) Update the modelPath variable below to the path of myModel.pt.
:: 2) Update the sourceLanguage and targetLanguage variables to those the model uses.
:: Use two letter language codes from: www.loc.gov/standards/iso639-2/php/code_list.php
:: 3) Place the word lists for that model in the same folder as myModel.pt, a.k.a. the modelPath directory.
:: For example, for converting a JPN->ENG model, the following files must exist in the modelPath directory:
:: dict.ja.txt
:: dict.en.txt
:: The conversion tool will prepend dict. and append .txt to the source and target languages to find them.
:: 3) If there is a vocabulary model file for the source language, update sourceLanguageVocabularyModel with the
:: absolute path to that vocabulary file. This is optional but recommended to ensure the best quality and speed from CTranslate2.
:: Alternatively, if any of the vocabModelSearchPath locations (relative to modelPath) contain the model and the Prefix and Postfix
:: are specified, then there will be an attempt to search for the vocabulary models using that information.
:: It is assumed that the source and destination vocabulary models are in the same folder. 
:: 4) Update pythonExe and converterExe to point to the correct directories. Be sure to include the literal binary name.
:: Example for portable versions:
:: set pythonExe=Python310\python.exe
:: set converterExe=Python310\Scripts\ct2-fairseq-converter.exe
:: 5) Save any changes to this file.
:: 6) Double-click on this .bat file to run it. Alternatively, open a command prompt, and then run this batch file.

:: Set defaults.
set modelPath=D:\myModel\big.pretrain.pt
set destinationPath=myModel_ctranslate2

:: Two letter language codes.
set sourceLanguage=ja
set targetLanguage=en

set pythonExe=python.exe
set converterExe=ct2-fairseq-converter.exe
set installedAsGlobal=false

:: set sourceLanguageVocabularyModel=invalid
set sourceLanguageVocabularyModel=invalid

:: if sourceLanguageVocabularyModel=invalid, then search for the vocab file using Prefix and Postfix specified.
set vocabularyModelPrefix=spm.
set vocabularyModelPostfix=.nopretok.model
set vocabularyModelPostfix2=.nopretok.vocab

:: These are relative to modelPath. Do not use a trailing \
set vocabModelSearchPath0=.
set vocabModelSearchPath1=spm
set vocabModelSearchPath2=spmModels
set vocabModelSearchPath3=..\spm
set vocabModelSearchPath4=..\spmModels


:: Do not modify the stuff below this line. ::


if /i "" neq "%~1" goto usage
:: Debug code.
:: %~dp0 returns a backslash \ at the end
::set currentDir=%cd%
:: sometimes %cd% will append a \, but sometimes not, so remove and then add it back to obtain a known state.
::if /i "%currentDir:~-1%" equ "\" set currentDir=%currentDir:~,-1%
::set currentDir=%currentDir%\
if /i "%destinationPath:~-1%" equ "\" set destinationPath=%destinationPath:~,-1%

:: Make all paths specified in the script, relative to the script instead of relative to current location of wherever command prompt happens to be.
pushd "%~dp0"

if not exist "%modelPath%" goto usage

:: Install required dependencies. This has an extremely high chance of failure because scikit-learn may require compiling.
python.exe -m pip install ctranslate2 fairseq scikit-learn
:: ctranslate2 dependencies: setuptools, numpy, pyyaml. Tested using: ctranslate2==3.24.0 setuptools==65.5.0 numpy==1.26.3 pyyaml==6.0.1
:: scikit-learn dependencies: hreadpoolctl, scipy, joblib. Tested using: scikit-learn==1.4.0 threadpoolctl==3.2.0 scipy==1.12.0 joblib==1.3.2

:: So, ct2-fairseq-converter.exe is not fully standalone. It must be run as python.exe ct2-fairseq-converter.exe because it looks for PyTorch in the environmental path. If python.exe is not specified, it seems to randomly add one under the current user account? Anyways, always preprend python.exe to prevent this strange behavior. This also means that this script has a hard dependency on being run where python.exe is available and also has both PyTorch and fairseq installed.
if /i "%installedAsGlobal%" neq "true" if not exist "%pythonExe%" (
echo  Error: Unable to find python.exe.
goto end)
if /i "%installedAsGlobal%" neq "true" if not exist "%converterExe%" (
echo  Error: Unable to find converterExe at: "%converterExe%"
goto end)
set converterExe="%pythonExe%" "%converterExe%"

set dataDir=invalid
call :determineDataDir "%modelPath%"
if not exist "%dataDir%" goto invalidDataDir


set vocabModelValid=False
if exist "%sourceLanguageVocabularyModel%" set vocabModelValid=True
if not exist "%sourceLanguageVocabularyModel%" call :updateSourceLanguageVocabModel

set config=--model_path "%modelPath%" --data_dir "%dataDir%" --output_dir "%destinationPath%" --force --source_lang %sourceLanguage% --target_lang %targetLanguage%

:: This appears to be incorrect, so comment it out. The correct way to generate a vocab_mapping file is at:
:: https://github.com/OpenNMT/papers/tree/master/WNMT2018/vmap
::if /i "%vocabModelValid%" equ "True" set config=%config% --vocab_mapping "%sourceLanguageVocabularyModel%"
:: The model will need to be re-converted once the correct vmap has been generated.

:: heuristic to see if core logic has already run
if exist "%destinationPath%\model.bin" goto afterCoreLogic

:: Core logic.
echo %converterExe% %config%
%converterExe% %config%

:afterCoreLogic


:: Copy sentencepiece models to destinationPath\spm.
:: First, check if the sentence pieces are available.
if /i "%vocabModelValid%" neq "True" (echo Warning: Unable to find sentencepiece models.
goto end)

:: The previous operation to convert might have failed and the output folder was never created. goto end if that was the case.
if not exist "%destinationPath%" goto end

:: Debug code.
::echo cd=%cd%

:: The copy command needs the directory already created.
if not exist "%destinationPath%\spm" mkdir "%destinationPath%\spm"

::if not exist target, then copy from source to target.
echo if not exist "%destinationPath%\spm\%sourceLanguageVocabModelName%" copy "%sourceLanguageVocabularyModel%" "%destinationPath%\spm\%sourceLanguageVocabModelName%"
if not exist "%destinationPath%\spm\%sourceLanguageVocabModelName%" copy "%sourceLanguageVocabularyModel%" "%destinationPath%\spm\%sourceLanguageVocabModelName%"
echo if not exist "%destinationPath%\spm\%sourceLanguageVocabModelName2%" "%sourceLanguageVocabularyModel2%" copy "%sourceLanguageVocabularyModel2%" "%destinationPath%\spm\%sourceLanguageVocabModelName2%"
if not exist "%destinationPath%\spm\%sourceLanguageVocabModelName2%" if exist "%sourceLanguageVocabularyModel2%" copy "%sourceLanguageVocabularyModel2%" "%destinationPath%\spm\%sourceLanguageVocabModelName2%"

echo if not exist "%destinationPath%\spm\%targetLanguageVocabModelName%" if exist "%targetLanguageVocabularyModel%" copy "%targetLanguageVocabularyModel%" "%destinationPath%\spm\%targetLanguageVocabModelName%"
if not exist "%destinationPath%\spm\%targetLanguageVocabModelName%" if exist "%targetLanguageVocabularyModel%" copy "%targetLanguageVocabularyModel%" "%destinationPath%\spm\%targetLanguageVocabModelName%"
echo if not exist "%destinationPath%\spm\%targetLanguageVocabModelName2%" if exist "%targetLanguageVocabularyModel2%" copy "%targetLanguageVocabularyModel2%" "%destinationPath%\spm\%targetLanguageVocabModelName2%"
if not exist "%destinationPath%\spm\%targetLanguageVocabModelName2%" if exist "%targetLanguageVocabularyModel2%" copy "%targetLanguageVocabularyModel2%" "%destinationPath%\spm\%targetLanguageVocabModelName2%"

goto end
:: Start functions list

:: Determines dataDir based upon input of first argument (%~1).
:determineDataDir
set dataDir=%~dp1
if /i "%dataDir:~-1%" equ "\" set dataDir=%dataDir:~,-1%

goto :eof

:: Goal is to fill sourceLanguageVocabularyModel with the correct path based upon vocabularyModelPrefix + sourceLanguage + vocabularyModelPostfix
:updateSourceLanguageVocabModel
if /i "%sourceLanguage%" equ "" goto :eof
if /i "%vocabularyModelPrefix%" equ "" goto :eof
if /i "%vocabularyModelPostfix%" equ "" goto :eof
set sourceLanguageVocabModelName=%vocabularyModelPrefix%%sourceLanguage%%vocabularyModelPostfix%
set sourceLanguageVocabModelName2=%vocabularyModelPrefix%%sourceLanguage%%vocabularyModelPostfix2%
set targetLanguageVocabModelName=%vocabularyModelPrefix%%targetLanguage%%vocabularyModelPostfix%
set targetLanguageVocabModelName2=%vocabularyModelPrefix%%targetLanguage%%vocabularyModelPostfix2%


:: stupid and simple way
::if exist "%dataDir%\%vocabModelSearchPath0%\%sourceLanguageVocabModelName%" set sourceLanguageVocabularyModel=%vocabModelSearchPath0%\%sourceLanguageVocabModelName%

:: smart and complicated way
for /l %%i in (4,-1,0) do if exist "%dataDir%\!vocabModelSearchPath%%i!\%sourceLanguageVocabModelName%" (
set sourceLanguageVocabularyModel=%dataDir%\!vocabModelSearchPath%%i!\%sourceLanguageVocabModelName%
set sourceLanguageVocabularyModel2=%dataDir%\!vocabModelSearchPath%%i!\%sourceLanguageVocabModelName2%
set targetLanguageVocabularyModel=%dataDir%\!vocabModelSearchPath%%i!\%targetLanguageVocabModelName%
set targetLanguageVocabularyModel2=%dataDir%\!vocabModelSearchPath%%i!\%targetLanguageVocabModelName2%
set vocabModelValid=True
)

:: Debug code
::echo set vocabModelValid=%vocabModelValid%
::echo set sourceLanguageVocabularyModel=%sourceLanguageVocabularyModel%
::echo set sourceLanguageVocabularyModel2=%sourceLanguageVocabularyModel2%
::echo set targetLanguageVocabularyModel=%targetLanguageVocabularyModel%
::echo set targetLanguageVocabularyModel2=%targetLanguageVocabularyModel2%
::set vocabModelValid=False

goto :eof


:invalidDataDir
echo  Error: Conversion failed.
echo  Reason: Unable to determine correct data directory based on: 
echo  modelPath="%modelPath%"
goto end


:usage
start notepad "%~0"


:end
popd
endlocal
pause
