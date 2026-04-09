#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Description:
A plugin library to add support for fairseq.sugoi to py3translationServer.

Usage:
import resources.engines.ctranslate2.sugoi as engine
or
engine = importlib.import_module( resources.engines.ctranslate2.sugoi )
or
sys.path.append( str( pathlib.Path( __file__ ).parent ) )
import sugoi as engine
And finally:
translator = engine.Translator( name='sugoiv4', modelPath=my/path/to/model.bin, modelSettings={ } )
print( translator.translate( [ 'a', 'list', 'of', 'strings' ] )

Requirements:
- python -m pip install sentencepiece
- fairseq is also required and the old versions on pypi.org/pip do not work. It must be compiled from source. Alternatively, use these wheels: https://github.com/gdiaz384/fairseq/releases
    python -m pip install fairseq-downloaded.whl
- CUDA requires a PyTorch build compatible with the local OS, hardware, and OS GPU driver. https://pytorch.org/get-started/previous-versions/ 
    For CUDA 11, use PyTorch <= 2.3.x
    For CUDA 12, use PyTorch >= 2.5.x
    Tested working on PyTorch 2.3.1 CUDA 11 + CPU
- DirectML requires selling your soul to microsoft. https://pypi.org/project/torch-directml/

Optional:
python -m pip install tensorboardX

Fairseq source code and documentation:
https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/__init__.py
https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/fairseq_model.p
https://github.com/facebookresearch/fairseq/blob/d13e14a800bb588e5a77fb4e551f554ff9b24a72/fairseq/models/fairseq_model.py#L242
https://github.com/facebookresearch/fairseq/blob/d13e14a800bb588e5a77fb4e551f554ff9b24a72/fairseq/models/fairseq_model.py#L287
https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_base.py
https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_legacy.py

https://fairseq.readthedocs.io/en/latest/models.html#fairseq.models.transformer.TransformerModel
https://fairseq.readthedocs.io/en/latest/_modules/fairseq/models/fairseq_model.html#BaseFairseqModel.from_pretrained
https://fairseq.readthedocs.io/en/latest/_modules/fairseq/tasks/translation.html?highlight=source_lang
https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-interactive

https://github.com/gdiaz384/py3TranslateLLM/wiki/fairseq-Installation-Guide

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
devices = ( 'cpu', 'cuda', 'gpu', 'directml' ) # gpu is aliased to cuda.
# Use two letter language codes: www.loc.gov/standards/iso639-2/php/code_list.php
defaultSourceLanguage = 'ja'
defaultTargetLanguage = 'en'
defaultModelBinNames = [ 'big.pretrain.pt' ]
consoleEncoding = 'utf-8'
debug = False

# These are internal variable names for fairseq and CTranslate2, so they use a slightly different variable naming scheme.
# https://fairseq.readthedocs.io/en/latest/_modules/fairseq/models/transformer/transformer_legacy.html#TransformerModel
# As a point of reference, OpenNMT refers to the text preprocessing as as the tokenizer type depending upon model or model format used: byte_bpe, bytes, characters, fastbpe, gpt2, bert, hf_byte_bpe, sentencepiece, subword_nmt
# Fairseq has tokenizer and bpe both available to set. The tokenizer can be: moses, space. 
# default_bpe uses the bpe value options defined by the fairseq API: (sentencepiece), fastbpe, subword_nmt
# Example sentence pieces: https://huggingface.co/JustFrederik
default_bpe = 'sentencepiece'
# sentencePieceModelFolders is where to search for the sentence piece models. The paths in sentencePieceModelFolders are relative to modelPath, which must be specified at runtime.
# If no sourceLanguageSpm is specified, then use the values in sentencePieceModelFolder together with defaultSourceLanguage to compute a value for sourceLanguageSpm and check if it exists as a file. If it exists, use it. Example:
#'spm/spm.ja.nopretok.model'
# If no targetLanguageSpm is specified, then use the values in sentencePieceModelFolder together with defaultTargetLanguage to compute a value for targetLanguageSpm and check if it exists as a file. If it exists, use it. Example:
#'../spmModel/spm.en.nopretok.model'
sentencePieceModelFolder = [ 'spm', 'spmModel', 'spmModels' ]
sentencePieceModelPrefix = 'spm.'
sentencePieceModelPostfix = '.nopretok.model'
# https://fairseq.readthedocs.io/en/latest/_modules/fairseq/tasks/fairseq_task.html?highlight=beam_size
# https://github.com/facebookresearch/fairseq/blob/d13e14a800bb588e5a77fb4e551f554ff9b24a72/fairseq/models/fairseq_model.py#L174C20-L174C20
# beam_size is the number of tokens generated by the model. The best one will be chosen as the return value. Directly affects quality. This is the main speed vs quality setting. Integer, Default=( 5 )
default_beam_size = 5
# Integer, ( 0 )
default_max_len_a = 0
# Integer, ( 200 )
default_max_len_b = 200
# Integer, ( 1 )
default_min_len = 1
# (True), False
default_normalize_scores = True
# Integer, ( 1 )
default_len_penalty = 1
# Integer, ( 0 )
default_unk_penalty = 0
# Float, ( 1.0 )
default_temperature = 1.0
# True, (False)
default_match_source_len = False
# Prevent repetitions of ngrams with this size (set 0 to disable). Integer, ( 0 )
default_no_repeat_ngram_size = 3


# Fairseq really likes inheritance which makes their source code very hard/impossible to read since one class is split up into 4+ files for the class + additional files for the config. The legacy code is also using argparse for configuration which makes their runtime parameters very hard to determine. The parameters do not all show up when using the inspect module.
# torch.nn.modules.module.Module as nn.Module -> BaseFairseqModel -> FairseqEncoderDecoderModel -> TransformerModelBase ->  TransformerModel

# import inspect
# inspect.signature( function )
# inspect.getfullargspec()
# getMethods( class ) # This is defined below.

# nn.Module has .cuda(), .cpu(), and .to() methods and inherents from <class 'object'>
# <Signature (*args, **kwargs) -> None>
# getMethods( torch.nn.modules.module.Module )
# ['T_destination', '__annotations__', '__call__', '__getattr__', '__getstate__','__setstate__', '_apply', '_call_impl', '_compiled_call_impl', '_get_backward_hooks', '_get_backward_pre_hooks', '_get_name', '_load_from_state_dict', '_maybe_warn_non_full_backward_hook', '_named_members', '_register_load_state_dict_pre_hook', '_register_state_dict_hook', '_replicate_for_data_parallel', '_save_to_state_dict', '_slow_forward', '_version', '_wrapped_call_impl', 'add_module', 'apply', 'bfloat16', 'buffers', 'call_super_init', 'children', 'compile', 'cpu', 'cuda', 'double', 'dump_patches', 'eval', 'extra_repr', 'float', 'forward', 'get_buffer', 'get_extra_state', 'get_parameter', 'get_submodule', 'half', 'ipu', 'load_state_dict', 'modules', 'named_buffers', 'named_children', 'named_modules', 'named_parameters', 'parameters', 'register_backward_hook', 'register_buffer', 'register_forward_hook', 'register_forward_pre_hook', 'register_full_backward_hook', 'register_full_backward_pre_hook', 'register_load_state_dict_post_hook', 'register_module', 'register_parameter', 'register_state_dict_pre_hook', 'requires_grad_', 'set_extra_state', 'share_memory', 'state_dict', 'to', 'to_empty', 'train', 'type', 'xpu', 'zero_grad']

# fairseq.models.BaseFairseqModel, <class 'fairseq.models.fairseq_model.BaseFairseqModel'>
# fairseq.models.fairseq_model.BaseFairseqModel.__bases__  # (<class 'torch.nn.modules.module.Module'>,)
# <Signature ()>
# FullArgSpec(args=['self'], varargs=None, varkw=None, defaults=None, kwonlyargs=[], kwonlydefaults=None, annotations={})
# getMethods( fairseq.models.BaseFairseqModel )
# ['T_destination', '__annotations__', '__call__', '__getattr__', '__getstate__','__setstate__', '_apply', '_call_impl', '_compiled_call_impl', '_get_backward_hooks', '_get_backward_pre_hooks', '_get_name', '_load_from_state_dict', '_maybe_warn_non_full_backward_hook', '_named_members', '_register_load_state_dict_pre_hook', '_register_state_dict_hook', '_replicate_for_data_parallel', '_save_to_state_dict', '_slow_forward', '_version', '_wrapped_call_impl', 'add_args', 'add_module', 'apply', 'bfloat16', 'buffers', 'build_model', 'call_super_init', 'children', 'compile', 'cpu', 'cuda', 'double', 'dump_patches', 'eval', 'extra_repr', 'extract_features', 'float', 'forward', 'from_pretrained', 'get_buffer', 'get_extra_state', 'get_normalized_probs', 'get_normalized_probs_scriptable', 'get_parameter', 'get_submodule', 'get_targets', 'half', 'hub_models', 'ipu', 'load_state_dict', 'make_generation_fast_', 'max_positions', 'modules', 'named_buffers', 'named_children', 'named_modules', 'named_parameters', 'parameters', 'prepare_for_inference_', 'prepare_for_onnx_export_', 'register_backward_hook', 'register_buffer', 'register_forward_hook', 'register_forward_pre_hook', 'register_full_backward_hook', 'register_full_backward_pre_hook', 'register_load_state_dict_post_hook', 'register_module', 'register_parameter', 'register_state_dict_pre_hook', 'requires_grad_', 'set_epoch', 'set_extra_state', 'set_num_updates', 'share_memory','state_dict', 'to', 'to_empty', 'train', 'type', 'upgrade_state_dict', 'upgrade_state_dict_named', 'xpu', 'zero_grad']
# fairseq.models.BaseFairseqModel.from_pretrained()
# <Signature (model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', **kwargs)>
# FullArgSpec(args=['cls', 'model_name_or_path', 'checkpoint_file', 'data_name_or_path'], varargs=None, varkw='kwargs', defaults=('model.pt', '.'), kwonlyargs=[], kwonlydefaults=None, annotations={})

# fairseq.models.FairseqEncoderDecoderModel, <class 'fairseq.models.fairseq_model.FairseqEncoderDecoderModel'>
# fairseq.models.fairseq_model.FairseqEncoderDecoderModel.__base__  # <class 'fairseq.models.fairseq_model.BaseFairseqModel'>
# <Signature (encoder, decoder)>
# FullArgSpec(args=['self', 'encoder', 'decoder'], varargs=None, varkw=None, defaults=None, kwonlyargs=[], kwonlydefaults=None, annotations={})
# getMethods( fairseq.models.FairseqEncoderDecoderModel )
# ['T_destination', '__annotations__', '__call__', '__getattr__', '__getstate__','__setstate__', '_apply', '_call_impl', '_compiled_call_impl', '_get_backward_hooks', '_get_backward_pre_hooks', '_get_name', '_load_from_state_dict', '_maybe_warn_non_full_backward_hook', '_named_members', '_register_load_state_dict_pre_hook', '_register_state_dict_hook', '_replicate_for_data_parallel', '_save_to_state_dict', '_slow_forward', '_version', '_wrapped_call_impl', 'add_args', 'add_module', 'apply', 'bfloat16', 'buffers', 'build_model', 'call_super_init', 'children', 'compile', 'cpu', 'cuda', 'double', 'dump_patches', 'eval', 'extra_repr', 'extract_features', 'float', 'forward', 'forward_decoder', 'from_pretrained', 'get_buffer', 'get_extra_state', 'get_normalized_probs', 'get_normalized_probs_scriptable', 'get_parameter', 'get_submodule', 'get_targets', 'half', 'hub_models', 'ipu', 'load_state_dict', 'make_generation_fast_', 'max_decoder_positions', 'max_positions', 'modules', 'named_buffers', 'named_children', 'named_modules', 'named_parameters', 'output_layer', 'parameters', 'prepare_for_inference_', 'prepare_for_onnx_export_', 'register_backward_hook', 'register_buffer', 'register_forward_hook', 'register_forward_pre_hook', 'register_full_backward_hook', 'register_full_backward_pre_hook', 'register_load_state_dict_post_hook', 'register_module', 'register_parameter', 'register_state_dict_pre_hook', 'requires_grad_', 'set_epoch', 'set_extra_state', 'set_num_updates', 'share_memory', 'state_dict', 'to','to_empty', 'train', 'type', 'upgrade_state_dict', 'upgrade_state_dict_named', 'xpu', 'zero_grad']
# fairseq.models.FairseqEncoderDecoderModel.from_pretrained()
# <Signature (model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', **kwargs)>
# FullArgSpec(args=['cls', 'model_name_or_path', 'checkpoint_file', 'data_name_or_path'], varargs=None, varkw='kwargs', defaults=('model.pt', '.'), kwonlyargs=[], kwonlydefaults=None, annotations={})

# fairseq.models.transformer.TransformerModelBase, <class 'fairseq.models.transformer.transformer_base.TransformerModelBase'>
# fairseq.models.transformer.transformer_base.TransformerModelBase.__bases__  # (<class 'fairseq.models.fairseq_model.FairseqEncoderDecoderModel'>,)
# <Signature (cfg, encoder, decoder)>
# FullArgSpec(args=['self', 'cfg', 'encoder', 'decoder'], varargs=None, varkw=None, defaults=None, kwonlyargs=[], kwonlydefaults=None, annotations={})
# getMethods( fairseq.models.transformer.TransformerModelBase )
# ['T_destination', '__annotations__', '__call__', '__getattr__', '__getstate__','__setstate__', '_apply', '_call_impl', '_compiled_call_impl', '_get_backward_hooks', '_get_backward_pre_hooks', '_get_name', '_load_from_state_dict', '_maybe_warn_non_full_backward_hook', '_named_members', '_register_load_state_dict_pre_hook', '_register_state_dict_hook', '_replicate_for_data_parallel', '_save_to_state_dict', '_slow_forward', '_version', '_wrapped_call_impl', 'add_args', 'add_module', 'apply', 'bfloat16', 'buffers', 'build_decoder', 'build_embedding', 'build_encoder', 'build_model', 'call_super_init', 'children', 'compile', 'cpu', 'cuda', 'double', 'dump_patches', 'eval', 'extra_repr', 'extract_features', 'float','forward', 'forward_decoder', 'from_pretrained', 'get_buffer', 'get_extra_state', 'get_normalized_probs', 'get_normalized_probs_scriptable', 'get_parameter', 'get_submodule', 'get_targets', 'half', 'hub_models', 'ipu', 'load_state_dict', 'make_generation_fast_', 'max_decoder_positions', 'max_positions', 'modules', 'named_buffers', 'named_children', 'named_modules', 'named_parameters', 'output_layer', 'parameters', 'prepare_for_inference_', 'prepare_for_onnx_export_', 'register_backward_hook', 'register_buffer', 'register_forward_hook', 'register_forward_pre_hook', 'register_full_backward_hook', 'register_full_backward_pre_hook', 'register_load_state_dict_post_hook', 'register_module', 'register_parameter', 'register_state_dict_pre_hook', 'requires_grad_', 'set_epoch', 'set_extra_state', 'set_num_updates', 'share_memory', 'state_dict', 'to', 'to_empty', 'train', 'type', 'upgrade_state_dict', 'upgrade_state_dict_named', 'xpu', 'zero_grad']
# fairseq.models.transformer.TransformerModelBase.from_pretrained()
# <Signature (model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', **kwargs)>

# https://fairseq.readthedocs.io/en/latest/_modules/fairseq/models/transformer/transformer_legacy.html#TransformerModel
# fairseq.models.transformer.TransformerModel, <class 'fairseq.models.transformer.transformer_legacy.TransformerModel'>
# fairseq.models.transformer.TransformerModel.__bases__  # (<class 'fairseq.models.transformer.transformer_base.TransformerModelBase'>,)
# <Signature (args, encoder, decoder)>
# FullArgSpec(args=['self', 'args', 'encoder', 'decoder'], varargs=None, varkw=None, defaults=None, kwonlyargs=[], kwonlydefaults=None, annotations={})
# getMethods( fairseq.models.transformer.TransformerModel )
# ['T_destination', '__annotations__', '__call__', '__dataclass', '__getattr__', '__getstate__', '__setstate__', '_apply', '_call_impl', '_compiled_call_impl', '_get_backward_hooks', '_get_backward_pre_hooks', '_get_name', '_load_from_state_dict', '_maybe_warn_non_full_backward_hook', '_named_members', '_register_load_state_dict_pre_hook', '_register_state_dict_hook', '_replicate_for_data_parallel', '_save_to_state_dict', '_slow_forward', '_version', '_wrapped_call_impl', 'add_args', 'add_module', 'apply', 'bfloat16', 'buffers', 'build_decoder', 'build_embedding', 'build_encoder', 'build_model', 'call_super_init', 'children', 'compile', 'cpu', 'cuda', 'double', 'dump_patches', 'eval', 'extra_repr', 'extract_features', 'float', 'forward', 'forward_decoder', 'from_pretrained', 'get_buffer', 'get_extra_state', 'get_normalized_probs', 'get_normalized_probs_scriptable', 'get_parameter', 'get_submodule', 'get_targets', 'half', 'hub_models', 'ipu', 'load_state_dict', 'make_generation_fast_', 'max_decoder_positions', 'max_positions','modules', 'named_buffers', 'named_children', 'named_modules', 'named_parameters', 'output_layer', 'parameters', 'prepare_for_inference_', 'prepare_for_onnx_export_', 'register_backward_hook', 'register_buffer', 'register_forward_hook', 'register_forward_pre_hook', 'register_full_backward_hook', 'register_full_backward_pre_hook', 'register_load_state_dict_post_hook', 'register_module', 'register_parameter', 'register_state_dict_pre_hook', 'requires_grad_', 'set_epoch', 'set_extra_state', 'set_num_updates', 'share_memory', 'state_dict', 'to', 'to_empty','train', 'type', 'upgrade_state_dict', 'upgrade_state_dict_named', 'xpu', 'zero_grad']
# fairseq.models.transformer.TransformerModel.from_pretrained()
# <Signature (model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', **kwargs)>
# FullArgSpec(args=['cls', 'model_name_or_path', 'checkpoint_file', 'data_name_or_path'], varargs=None, varkw='kwargs', defaults=('model.pt', '.'), kwonlyargs=[], kwonlydefaults=None, annotations={})

# fairseq.models.transformer.TransformerModel.from_pretrained() calls fairseq.checkpoint_utils.load_model_ensemble_and_task()


import sys
import os
import pathlib

import fairseq
#import ctranslate2
#import sentencepiece
#import torch
#import torch_directml
#try:
#    import psutil
#    psutilAvailable = True
#except:
#    psutilAvailable = False
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


"""
import inspect
# https://stackoverflow.com/questions/4241171/inspect-python-class-attributes
def getMethods( aClass ):
    dummy = dir( type( 'dummy', ( object, ), { } ) )
    items = [ ]
    for item in inspect.getmembers( aClass ):
        if ( item[ 0 ] in dummy ) == False:
            items.append( item[ 0 ] )
    return items
"""


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
        print( 'modelPath', modelPath )
        print( 'self.modelPath', self.modelPath )
        print( 'self.modelFolderOnly', self.modelFolderOnly )
        if device != None:
            self.device = device
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
        tempVersion = fairseq.__version__.split( '.' ) # These are all still strings.
        self.engineMajorVersion = int( tempVersion[ 0 ] )
        self.engineMinorVersion = int( tempVersion[ 1 ] )
        self.enginePatchVersion = int( tempVersion[ 2 ] )

        if 'sourceLanguage' in modelSettings:
            self.sourceLanguage = modelSettings[ 'sourceLanguage' ]
        else:
            self.sourceLanguage = models[ self.modelName ][ 'sourceLanguages' ][ 0 ]
        if 'targetLanguage' in modelSettings:
            self.targetLanguage = modelSettings[ 'targetLanguage' ]
        else:
            self.targetLanguage = models[ self.modelName ][ 'targetLanguages' ][ 0 ]
        if 'bpe' in modelSettings:
            self.bpe = modelSettings[ 'bpe' ]
        else:
            self.bpe = default_bpe
        if 'beam_size' in modelSettings:
            self.beam_size = modelSettings[ 'beam_size' ]
        else:
            self.beam_size = default_beam_size
        if 'max_len_a' in modelSettings:
            self.max_len_a = modelSettings[ 'max_len_a' ]
        else:
            self.max_len_a = default_max_len_a
        if 'max_len_b' in modelSettings:
            self.max_len_b = modelSettings[ 'max_len_b' ]
        else:
            self.max_len_b = default_max_len_b
        if 'min_len' in modelSettings:
            self.min_len = modelSettings[ 'min_len' ]
        else:
            self.min_len = default_min_len
        if 'normalize_scores' in modelSettings:
            self.normalize_scores = modelSettings[ 'normalize_scores' ]
        else:
            self.normalize_scores = default_normalize_scores
        if 'len_penalty' in modelSettings:
            self.len_penalty = modelSettings[ 'len_penalty' ]
        else:
            self.len_penalty = default_len_penalty
        if 'unk_penalty' in modelSettings:
            self.unk_penalty = modelSettings[ 'unk_penalty' ]
        else:
            self.unk_penalty = default_unk_penalty
        if 'temperature' in modelSettings:
            self.temperature = modelSettings[ 'temperature' ]
        else:
            self.temperature = default_temperature
        if 'match_source_len' in modelSettings:
            self.match_source_len = modelSettings[ 'match_source_len' ]
        else:
            self.match_source_len = default_match_source_len
        if 'no_repeat_ngram_size' in modelSettings:
            self.no_repeat_ngram_size = modelSettings[ 'no_repeat_ngram_size' ]
        else:
            self.no_repeat_ngram_size = default_no_repeat_ngram_size

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
        #self.sourceLanguageTokenizer = sentencepiece.SentencePieceProcessor( self.sourceLanguageSpm )
        #self.targetLanguageTokenizer = sentencepiece.SentencePieceProcessor( self.targetLanguageSpm )

        # Initialize the model.
        # https://fairseq.readthedocs.io/en/latest/_modules/fairseq/tasks/fairseq_task.html?highlight=beam_size
        parameters = {
        'checkpoint_file' : pathlib.Path( self.modelPath ).name,
        'source_lang' : self.sourceLanguage,
        'target_lang' : self.targetLanguage,
        'bpe' : self.bpe,
        'sentencepiece_model' : self.sourceLanguageSpm,
        'beam' : self.beam_size,
        'max_len_a' : self.max_len_a,
        'max_len_b' : self.max_len_b,
        'min_len' : self.min_len,
        'unnormalized' : self.normalize_scores,
        'lenpen' : self.len_penalty,
        'unkpen' : self.unk_penalty,
        'temperature' : self.temperature,
        'match_source_len' : self.match_source_len,
        'no_repeat_ngram_size' : self.no_repeat_ngram_size,
        }

        # ** means unpack the dictionary to key=value pairs suitable for use as function keyword parameters. One * means to unpack the dictionary's keynames and use them as positional parameters, not the value of the keys.
        self.translator = fairseq.models.transformer.TransformerModel.from_pretrained( self.modelFolderOnly, **parameters )

        # Move the model to the correct device.
        if ( self.device == 'cuda' ) or ( self.device == 'gpu' ):
            self.translator.cuda()
        elif self.device == 'directml':
            import torch
            import torch_directml
            dml = torch_directml.device()
            self.translator.to( dml )


    def encode( self, rawText, sourceLanguage=None, targetLanguage=None ):
        return rawText


    def translate( self, rawText, sourceLanguage=None, targetLanguage=None ):
        assert( isinstance( rawText, list) )
        # Update the self.lastSourceLanguage and self.lastTargetLanguage
        # Fairseq seems to require the languages when the model is initialized, so there is no need to keep track of and allow changing them afterwards. There are some multi-language fairseq models, but it is also not clear what their UI looks like. It should be possible to support such models once their UI is discovered.
#        if sourceLanguage == None:
#            self.lastSourceLanguage = defaultSourceLanguage
#        else:
#            if ( self.lastSourceLanguage == None ) or ( self.lastSourceLanguage != sourceLanguage ):
#                self.lastSourceLanguage = sourceLanguage
#        if targetLanguage == None:
#            self.lastTargetLanguage = defaultTargetLanguage
#        else:
#            if ( self.lastTargetLanguage == None ) or ( self.lastTargetLanguage != targetLanguage ):
#                self.lastTargetLanguage = targetLanguage

        # Tokenize rawText.
        #encodedList = self.encode( rawText, sourceLanguage, targetLanguage )
        #translatedList= self.translator.translate( rawText )
        #return self.decode( translatedList, sourceLanguage, targetLanguage )
        return self.translator.translate( rawText )


    def decode( self, translatedList, sourceLanguage=None, targetLanguage=None ):
        return translatedList


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

