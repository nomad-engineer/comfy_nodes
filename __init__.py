# Imports from myCustomNodes
from .loader import loader, loaderPipeOutput, loaderPipeInput
from .prompt import promptPipeOutput, promptSimple
from .conditioning import conditioning
from .controlNet import controlNetStacker
from .lora import LoraStackerTrigger, LoraStackGetTriggerWords
from .multivalue import ComplexValueNode
from .promptFromTagFiles import prompt_from_tag_files
from .promptAssembley import prompt_assemble
from .modelDownloader import ModelDownloader

# Imports from myCustomNodes2
from .nodes2 import MyPipeAll, pipeAdd, pipeRead, pipeReadKeyword, Sequential_Image_Loader, \
      Sequential_Image_Loader_Trigger, control_net_stack_concat, conditioning_2, loader as loader2, loader_out, \
      hash_image, GetImageSize, loader_in

NODE_CLASS_MAPPINGS = {
         # From myCustomNodes
         "Loader": loader,
         "Loader Pipe Output": loaderPipeOutput,
         "Loader Pipe Input": loaderPipeInput,
         "Prompt Pipe Output": promptPipeOutput,
         "Prompt Simple": promptSimple,
         "Conditioning": conditioning,
         "ControlNet_stacker": controlNetStacker,
         "Lora_Stacker_Trigger": LoraStackerTrigger,
         "Lora_Stack_Get_Trigger_Words": LoraStackGetTriggerWords,
         "Multi_value_node": ComplexValueNode,
         "Prompt from Tag Files": prompt_from_tag_files,
         "Prompt Assemble": prompt_assemble,
         "Model Downloader": ModelDownloader,

         # From myCustomNodes2
         "My Pipe All": MyPipeAll,
         "Pipe Read": pipeRead,
         "Pipe Write": pipeAdd,
         "Pipe Read Keyword": pipeReadKeyword,
         "Sequential Image Loader": Sequential_Image_Loader,
         "Sequential Image Loader Trigger": Sequential_Image_Loader_Trigger,
         "Control Net Stack Concat": control_net_stack_concat,
         "Conditioning 2": conditioning_2,
         "Loader2": loader2,
         "Loader Out": loader_out,
         "Hash Image": hash_image,
         "Get Image Size": GetImageSize,
         "Loader In": loader_in
}

NODE_DISPLAY_NAME_MAPPINGS = {
      # From myCustomNodes
      "Loader": "Loader",
      "Loader Pipe Output": "Loader Pipe Output",
      "Loader Pipe Input": "Loader Pipe Input",
      "Prompt Pipe Output": "Prompt Pipe Output",
      "Prompt Simple": "Prompt Simple",
      "Conditioning": "Conditioning",
      "ControlNet_stacker": "ControlNet_stacker",
      "Lora_Stacker_Trigger": "Lora_Stacker_Trigger",
      "Lora_Stack_Get_Trigger_Words": "Lora_Stack_Get_Trigger_Words",
      "Model Downloader": "Model Downloader",

      # From myCustomNodes2
      "My Pipe All": "MyPipeAll",
      "Pipe Read": "PipeRead",
      "Pipe Read Keyword": "PipeReadKeyword",
      "Pipe Write": "PipeWrite",
      "Sequential Image Loader": "SequentialImageLoader",
      "Sequential Image Loader Trigger": "SequentialImageLoaderTrigger",
      "Control Net Stack Concat": "ControlNetStackConcat",
      "Conditioning 2": "Conditioning2",
      "Loader2": "Loader2",
      "Loader Out": "LoaderOut",
      "Hash Image": "Hash Image",
      "Get Image Size": "Get Image Size",
      "Loader In": "Loader In"
}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
