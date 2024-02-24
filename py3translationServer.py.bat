@echo off
pushd "%~dp0"
:: '::' means comment

:: The NMT engine used to process the model. Can be fairseq or ctranslate2
set mode=fairseq
::set mode=ctranslate2

:: The path to the model. No quotes.
:: For fairseq, include the full path and model name.
:: For ctranslate2, include only the path to model.bin but not the model name itself.
set modelPath=D:\model\big.pretrain.pt
::set modelPath=ct2_model

:: Use two letter language codes: www.loc.gov/standards/iso639-2/php/code_list.php
set sourceLanguage=ja
set targetLanguage=en

:: The path and file name for the source sentence piece model. No quotes.
set sourceSentencePieceModel=%modelPath%\spm\spm.%sourceLanguage%.nopretok.model

:: The path and file name for the target sentence piece model. No quotes.
set targetSentencePieceModel=%modelPath%\spm\spm.%targetLanguage%.nopretok.model

:: Valid values are: cpu, gpu, cuda, directml. gpu is aliased to cuda. directml requires fairseq.
set device=cpu

:: The path and file name for Python.
:: To use with a portable version of Python, prepend a custom path to python.exe
set pythonExe=python.exe

:: Less common options. Append to core logic as needed.
:: Specify the internet protocol address for the server. 0.0.0.0 means bind to all local addresses.
:: --address 0.0.0.0
:: Specify port to listen on. Associated with --address. Default=14366.
:: --port 14366
:: Disable performance metrics (time keeping).
:: --disablePerfMetrics
:: Preload the model for lower latency inferencing.
:: --preloadModel
:: Print more information.
:: --verbose
:: Print too much information.
:: --debug

:: To change internal fairseq or ctranslate2 variables, look at the defaults near the top of the .py file.


:: Core logic. Invoke server with the options specified above.
"%pythonExe%" py3translationServer.py %mode% "%modelPath%" -dev %device% --sourceLanguage %sourceLanguage% --targetLanguage %targetLanguage% -sspm "%sourceSentencePieceModel%" -tspm "%targetSentencePieceModel%"
popd
