#import torch
import numpy as np
#import os
#import sys
#import csv
import comfy.sd
#import json
import folder_paths
import typing as tg
#import datetime
#import io
#from server import PromptServer, BinaryEventTypes
#from nodes import common_ksampler
#from PIL import Image
#from PIL.PngImagePlugin import PngInfo
#from pathlib import Path

class loader:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):

        checkpoint_files = ["None"] + folder_paths.get_filename_list("checkpoints")
        vae_files = ["None"] + folder_paths.get_filename_list("vae")
        vae_options = ['baked', 'vae']

        return {
            "required": {
                "model_name": (checkpoint_files,),
                "vae_name": (vae_files,),
                "vae_source": (vae_options,),
                "clip_skip": ("INT",{ "default": -1, "min": -5, "max": -1, "step": 1}),
            },
            "optional": { 
            },
        }
 
    RETURN_TYPES = ("LOADER_PIPE", "MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("loader_pipe", "model", "clip", "vae")
 
    FUNCTION = "doit"
 
    #OUTPUT_NODE = False
 
    CATEGORY = "z_mynodes"
 
    def doit(self, model_name, vae_name, vae_source, clip_skip):
        if  model_name == "None":
            print(f"Select Model: No model selected")
            return()

        #
        ckpt_path = folder_paths.get_full_path("checkpoints", model_name)
        model, clip, vae, clipvision = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True,
                                                     embedding_directory=folder_paths.get_folder_paths("embeddings"))
        
        clip = clip.clone()
        clip.clip_layer(clip_skip)

        # Load VAE
        if vae_source == "baked":
            pass
        elif vae_source == "vae":
            if vae_name in ["taesd", "taesdxl", "taesd3"]:
                sd = self.load_taesd(vae_name)
            else:
                vae_path = folder_paths.get_full_path("vae", vae_name)
                sd = comfy.utils.load_torch_file(vae_path)
            vae = comfy.sd.VAE(sd=sd)


        pipe_line = model, clip, vae

        return (pipe_line, model, clip, vae,)


 
class loaderPipeOutput:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "loader_pipe": ("LOADER_PIPE",)
            }
        }
 
    RETURN_TYPES = ('LOADER_PIPE', 'MODEL', 'CLIP', 'VAE',)
    RETURN_NAMES = ('loader_pipe', 'model', 'clip', 'vae',)
 
    FUNCTION = "doit"
 
    #OUTPUT_NODE = False
 
    CATEGORY = "z_mynodes"
 
    def doit(self, loader_pipe):

        model, clip, vae = loader_pipe
        
        return (loader_pipe, model, clip, vae, )
    
class loaderPipeInput:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
            },
            "optional": { 
                "loader_pipe": ("LOADER_PIPE", ),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
            },
        }
 
    RETURN_TYPES = ("LOADER_PIPE", )
    RETURN_NAMES = ("loader_pipe", )
 
    FUNCTION = "doit"
 
    #OUTPUT_NODE = False
 
    CATEGORY = "z_mynodes"
 
    def doit(self, loader_pipe=None, model=None, clip=None, vae=None):
        if loader_pipe is not None:
            modelOut, clipOut, vaeOut = loader_pipe
        
        if model is not None:
            modelOut = model
        if clip is not None:
            clipOut = clip
        if vae is not None:
            vaeOut = vae

        pipe_line = modelOut, clipOut, vaeOut

        return (pipe_line, modelOut, clipOut, vaeOut,)
