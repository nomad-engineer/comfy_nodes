import os
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageChops, ImageFont
from PIL.PngImagePlugin import PngInfo
import torch
import numpy as np
import json
import hashlib
from server import PromptServer
import folder_paths
import comfy.sd
from nodes import ControlNetApplyAdvanced
from dataclasses import dataclass, field, fields
from typing import Any
from typing import Any, Dict
import math
import copy
import io
import random
from pathlib import Path, PurePath
import fnmatch
import random


# SHA-256 Hash
def get_sha256(file_path, output_size=None):
    """
    Calculate the SHA-256 hash of a file with an optional output size.

    :param file_path: Path to the file to hash.
    :param output_size: Number of characters of the hash to return (optional).
    :return: SHA-256 hash string, truncated if output_size is provided.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            sha256_hash.update(chunk)
    hash_hex = sha256_hash.hexdigest()
    if output_size is not None:
        return hash_hex[:output_size]
    return hash_hex

import hashlib

def generate_image_hash(tensor, output_size=None):
    """
    Generate a unique hash for a PyTorch tensor image with optional output length.

    :param tensor: A PyTorch tensor representing the image (C, H, W) or (N, C, H, W).
    :param output_size: The number of characters to include in the hash (optional).
    :return: A unique hash string for the image, truncated to the specified length if provided.
    """
    # Ensure the tensor is in CPU and detached (for safety)
    tensor = tensor.cpu().detach()
    
    # Convert tensor to bytes
    tensor_bytes = tensor.numpy().tobytes()

    # Generate the hash
    hash_value = hashlib.sha256(tensor_bytes).hexdigest()
    
    # Truncate the hash if an output size is specified
    if output_size is not None:
        return hash_value[:output_size]
    return hash_value


# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class AnyType(str):
    # Credit to pythongosssss    

    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")



def get_text_from_image_file(image_file):
    """
    Tries to open a corresponding .txt file for an image.
    1. Looks for a .txt file with the exact same name (e.g., image.png -> image.txt).
    2. If not found, looks for a .txt file with the base name (e.g., image_01.png -> image.txt).
    
    Args:
        image_file (str): The path to the image file.

    Returns:
        str: The content of the found text file, or an empty string if no file is found.
    """
    text = ""
    base_name = os.path.splitext(image_file)[0]
    
    # 1. Try to open the same named .txt file
    text_file_same_name = base_name + ".txt"
    if os.path.exists(text_file_same_name):
        with open(text_file_same_name, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        # 2. If that doesn't exist, look for the same named file without a suffix.
        # This assumes the base name without a suffix is what comes before the last '_' or a hyphen.
        # We'll try to find the part before the last non-alphanumeric character (like _ or -)
        # or simply the part before the last number sequence if it's like file_01.
        
        # Let's try a more robust approach to find the "root" part of the filename.
        # We'll split by common separators and take the first part.
        
        # Get just the filename from the path
        filename_only = os.path.basename(base_name)
        
        # Common separators to consider for "prefix"
        separators = ['_', '-']
        found_root = False
        for sep in separators:
            if sep in filename_only:
                parts = filename_only.split(sep)
                # Check if the last part is numeric or looks like an index
                if parts[-1].isdigit() or (len(parts) > 1 and parts[-1].isalnum() and not parts[-1].isalpha() and not parts[-1].isdigit()):
                    root_name = sep.join(parts[:-1])
                    text_file_root = os.path.join(os.path.dirname(image_file), root_name + ".txt")
                    if os.path.exists(text_file_root):
                        with open(text_file_root, "r", encoding="utf-8") as f:
                            text = f.read()
                        found_root = True
                        break
        
        # If no separator-based root was found, or it didn't lead to a file,
        # try a simpler approach by just removing the last numeric sequence if present.
        if not found_root:
            import re
            match = re.match(r"^(.*?)(?:[_-]?\d+)?$", filename_only)
            if match:
                root_name_simple = match.group(1)
                text_file_root_simple = os.path.join(os.path.dirname(image_file), root_name_simple + ".txt")
                if os.path.exists(text_file_root_simple):
                    with open(text_file_root_simple, "r", encoding="utf-8") as f:
                        text = f.read()

    return text

@dataclass
class Loader:
    # Initial fields with default values
    model: Any = field(default=None)
    clip: Any = field(default=None)
    vae: Any = field(default=None) 

class Conditioning:
    # Initial fields with default values
    loader: Any = field(default=None)
    controlnet_stack: Any = field(default=None)
    lora_stack: Any = field(default=None)
    latent: Any = field(default=None)
    image: Any = field(default=None)
    positive: Any = field(default=None)
    negative: Any = field(default=None)
    seed: Any = field(default=None)

@dataclass
class Pipe:
    # Initial fields with default values
    loader: Any = field(default_factory=Loader)
    controlNetStack: Any = field(default=None)
    loraStack: Any = field(default=None)
    imageInput: Any = field(default=None)
    imageProcessing: Any = field(default=None)
    imageUpscaled: Any = field(default=None)
    promptPositive: Any = field(default=None)
    promptNegative: Any = field(default=None) 
    seed: Any = field(default=None)

    def add_field(self, name: str, value):
        """
        Dynamically add a new FieldUpdater attribute to the Pipe instance.
        
        """

        # Add the new field to the instance dynamically
        setattr(self, name, value)

    def read_field(self, name: str):
        return getattr(self, name)
 

#----------Nodes-------------------------------------------------------------------    

class MyPipeAll:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                
            },
            "optional":{
                "myPipe": ("MY_PIPE",),
                "loader": ("LOADER_PIPE",),
                "controlNetStack": ("CONTROL_NET_STACK",),
                "loraStack": ("LORA_STACK",),
                "imageInput": ("IMAGE",),
                "imageProcessing": ("IMAGE",),
                "imageUpscaled": ("IMAGE",),
                "promptPositive": ("STRING", {'forceInput':True}),
                "promptNegative": ("STRING", {'forceInput':True}),
                "seed": ("INT", {'forceInput':True})
                },
        }

 
    RETURN_TYPES = ('MY_PIPE', 'LOADER_PIPE', 'CONTROL_NET_STACK', 'LORA_STACK', 'IMAGE', 'IMAGE', 'IMAGE', 'STRING', 'STRING', 'INT')
    RETURN_NAMES = ('*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*')
 
    FUNCTION = "doit"
 
    #OUTPUT_NODE = False
 
    CATEGORY = "my_nodes"
 
    def doit(self, myPipe=Pipe(), loader=None, controlNetStack=None, loraStack=None, imageInput=None, imageProcessing=None, imageUpscaled=None, promptPositive=None, promptNegative=None, seed=None):

        if loader is not None: myPipe.loader = loader
        if controlNetStack is not None: myPipe.controlNetStack = controlNetStack
        if loraStack is not None: myPipe.loraStack = loraStack
        if imageInput is not None: myPipe.imageInput = imageInput
        if imageProcessing is not None: myPipe.imageProcessing = imageProcessing
        if imageUpscaled is not None: myPipe.imageUpscaled = imageUpscaled
        if promptPositive is not None: myPipe.promptPositive = promptPositive
        if promptNegative is not None: myPipe.promptNegative = promptNegative
        if seed is not None: myPipe.seed = seed

        return (myPipe, myPipe.loader, myPipe.controlNetStack, myPipe.loraStack, myPipe.imageInput, myPipe.imageProcessing, myPipe.imageUpscaled, myPipe.promptPositive, myPipe.promptNegative, myPipe.seed)
    
class pipeAdd:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "key_1": ("STRING",),
                "key_2": ("STRING",),
            },
            "optional":{       
                "my_pipe": ("MY_PIPE",),
                "value_1": (any_type,),
                "value_2": (any_type,)
                }
        }
 
    RETURN_TYPES = ('MY_PIPE',)
    RETURN_NAMES = ('my_pipe',)
 
    FUNCTION = "doit"
 
    #OUTPUT_NODE = False
 
    CATEGORY = "my_nodes"
 
    def doit(self, key_1, key_2, value_1=None, value_2=None, my_pipe=Pipe()):
        if value_1 is not None: my_pipe.add_field(key_1, value_1)
        if value_2 is not None: my_pipe.add_field(key_2, value_2)

        return (my_pipe,)
    
class pipeReadKeyword:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "my_pipe": ("MY_PIPE",), 
                "key_1": ("STRING",),
                "key_2": ("STRING",),
            },
            "optional":{       
                
                }
        }
 
    RETURN_TYPES = ('MY_PIPE', any_type, any_type,)
    RETURN_NAMES = ('my_pipe', 'value_1', 'value_2',)
 
    FUNCTION = "doit"
 
    OUTPUT_NODE = False
 
    CATEGORY = "my_nodes"
 
    def doit(self, key_1, key_2, my_pipe:Pipe):
        value_1 = None
        value_2 = None

        if key_1 is not '': value_1 = my_pipe.read_field(key_1)
        if key_2 is not '': value_2 = my_pipe.read_field(key_2)

        return (my_pipe, value_1, value_2,)
    
class pipeRead:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):

        pipe_fields = [f.name for f in fields(Pipe)]

        return {
            "required":{
                "my_pipe": ("MY_PIPE",), 
                "key_1": (pipe_fields,),
                "key_2": (pipe_fields,),
            },
            "optional":{       
                
                }
        }
 
    RETURN_TYPES = ('MY_PIPE', any_type, any_type,)
    RETURN_NAMES = ('my_pipe', 'value_1', 'value_2',)
 
    FUNCTION = "doit"
 
    OUTPUT_NODE = False
 
    CATEGORY = "my_nodes"
 
    def doit(self, key_1, key_2, my_pipe:Pipe):

        value_1 = None
        value_2 = None
        if key_1 is not '': value_1 = my_pipe.read_field(key_1)
        if key_2 is not '': value_2 = my_pipe.read_field(key_2)

        return (my_pipe, value_1, value_2,)
    

class Sequential_Image_Loader:
    def __init__(self):
        # Initialize the image list and current index
        self.image_files = []
        self.current_index = 0
        self.current_image = ''
        self.current_path = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": '', "multiline": False}),
                "loop_through_images": ("BOOLEAN", {"default": False}),
                "regex": ("STRING", {"default": '*', "multiline": False}),

            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING", "STRING", "STRING", "STRING", "IMAGE_LOADER_TRIGGER",)
    RETURN_NAMES = ("image", "no_images_left", "text_from_file", "filename", "rel_path", "abs_path", "to trigger",)
    FUNCTION = "load_next_image"
    CATEGORY = "my_nodes"
    OUTPUT_NODE = False

    def _save_state(self, folder_path):
        """Save the current index to a JSON file in the folder."""
        state_file = os.path.join(folder_path, "processing.json")
        state = {
            "current_index": self.current_index,
            "current_image": self.current_path
        }
        try:
            with open(state_file, "w") as f:
                json.dump(state, f)
        except IOError as e:
            print(f"Error saving state to {state_file}: {e}")

    def _load_state(self, folder_path):
        """Load the current index from a JSON file in the folder."""
        state_file = os.path.join(folder_path, "processing.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    content = f.read().strip()
                    if content:
                        state = json.loads(content)
                        self.current_index = state.get("current_index", 0)
                    else:
                        self.current_index = 0
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading state from {state_file}: {e}")
                self.current_index = 0  # Reset index on error
        else:
            self.current_index = 0

    def load_next_image(self, path, loop_through_images, regex):
        self.path = path
        self.loop_through_images = loop_through_images

        # Load the previous state if it exists
        self._load_state(path)
        print(f"Current Index: {self.current_index}")
        print(f"Current Image: {self.current_path}")


        # If the path changes or the list is empty, reload images
        if not self.image_files or self.current_path != path:
            if not os.path.exists(path):
                raise ValueError(f"Path does not exist: {path}")

            folder_path = Path(path)
            file_type = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif")
            self.image_files = []
            for type in file_type:
                self.image_files.extend(folder_path.rglob(type))

            self.image_files = [
                f for f in self.image_files
                if fnmatch.fnmatch(str(f), regex)
                ] if regex != "*" else self.image_files

            if not self.image_files:
                raise ValueError(f"No valid image files found in the specified path: {path}")

            # Save the current path
            self.current_path = path

            print(f"Loaded {len(self.image_files)} images from path: {path}")

        # Check if there are images left to process
        self.images_left = len(self.image_files) - self.current_index
        has_images = self.images_left > 0

        # Restart index if loop enabled and images have run out
        if not has_images and loop_through_images:
            self.current_index = 0
            has_images = True
            print("Image index reset")
        elif not has_images and not loop_through_images:
            raise ValueError("No images left to process")

        # Load the next image and associated text if available
        if has_images:
            image_file = self.image_files[self.current_index]
            self.current_image = str(image_file)
            rel_path = image_file.relative_to(path).parent
            abs_path = (path / rel_path).resolve()
            image = Image.open(image_file).convert("RGB")  # Ensure image is in a compatible format

            # Extract the filename without extension
            filename = os.path.splitext(os.path.basename(image_file))[0]

            # Try to load the corresponding .txt file
            text = get_text_from_image_file(image_file)
            
            #text_file = os.path.splitext(image_file)[0] + ".txt"
            #if os.path.exists(text_file):
            #    with open(text_file, "r", encoding="utf-8") as f:
            #        text = f.read()
            #else:
            #    text = ""

        else:
            image = None
            text = ""
            filename = ""

        # Check if there are images left to process
        self.images_left = len(self.image_files) - self.current_index

        return (pil2tensor(image), self.images_left, text, filename, str(rel_path), str(abs_path), self,)
    
    
    #def IS_CHANGED(cls, **kwargs):
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        #if kwargs['mode'] != 'single_image':
        #    return float("NaN")
        #else:
        #    fl = Sequential_Image_Loader.load_next_image(kwargs['path'])
        #    filename = fl.get_current_image()
        #    image = os.path.join(kwargs['path'], filename)
        #    sha = get_sha256(image)
        #    return sha

        #regex = kwargs.get('regex', '')
        #path = kwargs.get('path', '')
        #loop_through_images = kwargs.get('loop_through_images', '')
        
        change_values = []
        change_values.append(str(random.random()))
        #change_values.append(regex)
        #change_values.append(path)
        #change_values.append(loop_through_images)

        #fl = Sequential_Image_Loader.load_next_image(path, loop_through_images, regex)
        #filename = fl.get_current_image()
        #image = os.path.join(kwargs['path'], filename)
        #change_values.append(get_sha256(image))

        return_value = tuple(change_values)
        print('return_value: {}'.format(return_value))
        return (float("NaN"),)

class Sequential_Image_Loader_Trigger:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "trigger": (any_type,),
                    "from_loader": ("IMAGE_LOADER_TRIGGER",),
                    }
                }

    FUNCTION = "doit"

    CATEGORY = "my_nodes"
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("trigger_pass_through",)
    OUTPUT_NODE = True

    def doit(self, trigger, from_loader,):

        otherself = from_loader
        # Update the index and save the state
        otherself.current_index = otherself.current_index + 1
       
        otherself._save_state(otherself.path)

        # Check if there are images left to process
        otherself.images_left = otherself.images_left - 1

        if otherself.images_left > 0 or otherself.loop_through_images:
            PromptServer.instance.send_sync("impact-add-queue", {})

        return (trigger,)
    

class control_net_stack_concat:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
            },
            "optional":{       
                "CONTROL_NET_STACK_1": ("CONTROL_NET_STACK",),
                "CONTROL_NET_STACK_2": ("CONTROL_NET_STACK",),
                "CONTROL_NET_STACK_3": ("CONTROL_NET_STACK",),
                "CONTROL_NET_STACK_4": ("CONTROL_NET_STACK",)
                }
        }
 
    RETURN_TYPES = ('CONTROL_NET_STACK',)
    RETURN_NAMES = ('CONTROL_NET_STACK',)
 
    FUNCTION = "doit"
 
    OUTPUT_NODE = False
 
    CATEGORY = "my_nodes"
 
    def doit(self, CONTROL_NET_STACK_1=None, CONTROL_NET_STACK_2=None, CONTROL_NET_STACK_3=None, CONTROL_NET_STACK_4=None):

        stack = []

        if CONTROL_NET_STACK_1 is not None:
            stack.extend(CONTROL_NET_STACK_1)
        
        if CONTROL_NET_STACK_2 is not None:
            stack.extend(CONTROL_NET_STACK_2)

        if CONTROL_NET_STACK_3 is not None:
            stack.extend(CONTROL_NET_STACK_3)

        if CONTROL_NET_STACK_4 is not None:
            stack.extend(CONTROL_NET_STACK_4)


        return (stack,)
    
class conditioning_2:
    
    aspects = [
        ["1:1 square 1024x1024", 1024, 1024],
        ["3:4 portrait 896x1152", 896, 1152],
        ["5:8 portrait 832x1216", 832, 1216],
        ["9:16 portrait 768x1344", 768, 1344],
        ["9:21 portrait 640x1536", 640, 1536],
        ["4:3 landscape 1152x896", 1152, 896],
        ["3:2 landscape 1216x832", 1216, 832],
        ["16:9 landscape 1344x768", 1344, 768],
        ["21:9 landscape 1536x640", 1536, 640],
    ]

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):

        aspect_ratios = [name for name, width, height in s.aspects]
                
        return {
            "required":{
                "latent_from": (['latent', 'image', 'blank', 'blank_random', 'blank_from_image'], {'default': 'blank'}),
                "latent_batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "aspect": (aspect_ratios,),
                "apply_controlnet": (['false', 'true'],{'default': 'false'}),
                "apply_lora": (['false', 'true'],{'default':'false'}),
                "seed_from": (['new', 'pipe'], {'default': 'new'}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True})
            },
            "optional":{
                "loader_pipe": ("LOADER_PIPE",),
                "conditioning_pipe": ("CONDITIONING_PIPE",),
                "controlnet_stack": ("CONTROL_NET_STACK", ),
                "lora_stack": ("LORA_STACK", ),
                "latent": ("LATENT",),
                "image": ("IMAGE",),
                "positive": ("STRING",),
                "negative": ("STRING",)
            },
        }
    
    RETURN_TYPES = ('CONDITIONING_PIPE', 'MODEL', 'CONDITIONING', 'CONDITIONING', 'LATENT', 'VAE', 'CLIP', 'IMAGE', 'INT')
    RETURN_NAMES = ('conditioning_pipe', 'model', 'positive', 'negative', 'latent', 'vae', 'clip', 'image', 'seed')
 
    FUNCTION = "doit"
 
    OUTPUT_NODE = False
 
    CATEGORY = "my_nodes"
 
    def doit(self, latent_from, latent_batch_size, aspect, apply_controlnet, apply_lora, seed_from, seed, loader_pipe=None, latent=None, image=None, controlnet_stack=None, lora_stack=None, positive=None, negative=None, conditioning_pipe=Conditioning()):
        
        # unpack pipe and update with inputs
        if loader_pipe is not None: conditioning_pipe.loader = loader_pipe
        if controlnet_stack is not None: conditioning_pipe.controlnet_stack = controlnet_stack
        if lora_stack is not None: conditioning_pipe.lora_stack = lora_stack
        if latent is not None: conditioning_pipe.latent = latent
        if image is not None: conditioning_pipe.image = image
        if positive is not None: conditioning_pipe.positive = positive
        if negative is not None: conditioning_pipe.negative = negative
        if seed_from == 'new': conditioning_pipe.seed = seed
        
        model = conditioning_pipe.loader.model
        clip = conditioning_pipe.loader.clip
        vae = conditioning_pipe.loader.vae

        #apply lora
        if conditioning_pipe.lora_stack is not None and apply_lora == 'true':
            
            # Loop through the stack
            for tup in conditioning_pipe.lora_stack:
                defaults = ('none', 1.0, 1.0, [])  # Default values for the tuple
                lora_name, strength_model, strength_clip = tup
                
                lora_path = folder_paths.get_full_path("loras", lora_name)
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                
                print(f"Applying lora {lora_name}")
                model, clip = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)  

        print("Applying positive conditioning_pipe")
        tokens = clip.tokenize(conditioning_pipe.positive)
        condPos, pooledPos = clip.encode_from_tokens(tokens, return_pooled=True)
        positive_cond = [[condPos, {"pooled_output": pooledPos}]]

        print("Applying negative conditioning_pipe")
        tokens = clip.tokenize(conditioning_pipe.negative)
        condNeg, pooledNeg = clip.encode_from_tokens(tokens, return_pooled=True)
        negative_cond = [[condNeg, {"pooled_output": pooledNeg}]]

        #apply controlnet
        if conditioning_pipe.controlnet_stack is not None and apply_controlnet == 'true':
            for controlnet_tuple in conditioning_pipe.controlnet_stack:
                controlnet_name, image_cnet, strength, start_percent, end_percent  = controlnet_tuple
                
                if type(controlnet_name) == str:
                    controlnet_path = folder_paths.get_full_path("controlnet", controlnet_name)
                    controlnet = comfy.controlnet.load_controlnet(controlnet_path)
                else:
                    controlnet = controlnet_name
                
                print(f"Applying controlnet {controlnet_name}")
                controlnet_conditioning = ControlNetApplyAdvanced().apply_controlnet(positive_cond, negative_cond,
                                                                                     controlnet, image_cnet, strength,
                                                                                     start_percent, end_percent)

                positive_cond, negative_cond = controlnet_conditioning[0], controlnet_conditioning[1]


        #get resolution
        if latent_from == "latent":
            latent_out = conditioning_pipe.latent

        elif latent_from == 'image':
            t = vae.encode(image[:,:,:,:3])
            latent_out = {"samples":t}

        elif latent_from == 'blank_random':
            noOptions = len(self.aspects)
            name, width, height = self.aspects[random.randint(0, noOptions - 1)]

        elif latent_from == "blank_from_image":
            width, height = conditioning_pipe.image.shape[2], conditioning_pipe.image.shape[1]

        else:
            for aspect_item in self.aspects:
                if aspect_item[0] == aspect:
                    width = aspect_item[1]
                    height = aspect_item[2]
                    continue

        if latent_from not in ['image', 'latent']:
            latent_out = {"samples":torch.zeros([1, 4, height // 8, width // 8])}

        #repeat latent batch
        print("Repeating latent batch")
        batch = latent_out.copy()
        batch_in = latent_out["samples"]
        
        batch["samples"] = batch_in.repeat((latent_batch_size, 1,1,1))
        if "noise_mask" in latent_out and latent_out["noise_mask"].shape[0] > 1:
            masks = latent_out["noise_mask"]
            if masks.shape[0] < batch_in.shape[0]:
                masks = masks.repeat(math.ceil(batch_in.shape[0] / masks.shape[0]), 1, 1, 1)[:batch_in.shape[0]]
            batch["noise_mask"] = latent_out["noise_mask"].repeat((latent_batch_size, 1,1,1))
        if "batch_index" in batch:
            offset = max(batch["batch_index"]) - min(batch["batch_index"]) + 1
            batch["batch_index"] = batch["batch_index"] + [x + (i * offset) for i in range(1, latent_batch_size) for x in batch["batch_index"]]


        return (
            conditioning_pipe,
            model,
            positive_cond, 
            negative_cond, 
            batch,
            vae,
            clip,
            image,
            seed,
        )
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs['mode'] != 'single_image':
            return float("NaN")
        else:
            fl = Sequential_Image_Loader.load_next_image(kwargs['path'])
            filename = fl.get_current_image()
            image = os.path.join(kwargs['path'], filename)
            sha = get_sha256(image)
            return sha
        

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
 
    RETURN_TYPES = ("LOADER_PIPE",)
    RETURN_NAMES = ("loader_pipe",)
 
    FUNCTION = "doit"
 
    OUTPUT_NODE = False
 
    CATEGORY = "my_nodes"
    def doit(self, model_name, vae_name, vae_source, clip_skip):
        
        if  model_name == "None":
            print(f"Select Model: No model selected")
            return()

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

        loader.model = model
        loader.clip = clip
        loader.vae = vae

        return (loader, loader.model, loader.clip, loader.vae,)

class loader_in:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "MODEL": ('MODEL',),
                "CLIP": ('CLIP',),
                "VAE": ('VAE',),
            },
            "optional": {
            },
        }
 
    RETURN_TYPES = ("LOADER_PIPE",)
    RETURN_NAMES = ("loader_pipe",)
 
    FUNCTION = "doit"
 
    OUTPUT_NODE = False
 
    CATEGORY = "my_nodes"
    def doit(self, MODEL, CLIP, VAE):
        
        loader = Loader()
        loader.model = MODEL
        loader.clip = CLIP
        loader.vae = VAE

        return (loader,)

class loader_out:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "loader": ('LOADER_PIPE',)
            },
            "optional": {
            },
        }
 
    RETURN_TYPES = ("LOADER", "MODEL", "VAE", "CLIP",)
    RETURN_NAMES = ("loader", "model", "vae", "clip",)
 
    FUNCTION = "doit"
 
    OUTPUT_NODE = False
 
    CATEGORY = "my_nodes"
    def doit(self, loader:Loader):

        return (loader, loader.model, loader.vae, loader.clip,)
    
class hash_image:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "image": ('IMAGE',),
                "size": ("INT",{ "default": 10, "min": 1, "max": 100, "step": 1})
            },
            "optional": {
            },
        }
 
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
 
    FUNCTION = "doit"
 
    OUTPUT_NODE = False
 
    CATEGORY = "my_nodes"
    def doit(self, image, size,):

        if image is list:
            sha = generate_image_hash(image[0], size)
        else:
            sha = generate_image_hash(image, size)

        return (sha, )

class GetImageSize:

    SDXL_SUPPORTED_RESOLUTIONS = [
    (1024, 1024, 1.0),
    (1152, 896, 1.2857142857142858),
    (896, 1152, 0.7777777777777778),
    (1216, 832, 1.4615384615384615),
    (832, 1216, 0.6842105263157895),
    (1344, 768, 1.75),
    (768, 1344, 0.5714285714285714),
    (1536, 640, 2.4),
    (640, 1536, 0.4166666666666667),
    ]
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "get": (['image size', 'nearest sdxl',], {'default': 'image size'}),
            "scale_by": ("FLOAT", {"default": 1, "min": 0.01, "max": 8.0, "step": 0.1})
        }}

    RETURN_TYPES = ("IMAGE","INT", "INT", "INT",)
    RETURN_NAMES = ("image", "width", "height", "count",)
    FUNCTION = "doit"
    CATEGORY = "my_nodes"
    DESCRIPTION = """
Returns width, height and batch size of the image,  
and passes it through unchanged.  

"""

    def doit(self, image, get, scale_by):
        if get == "image size":
            width, height = self.getsize(image)
        if get == "nearest sdxl":
            width, height = self.getNearestResolution(image)
        width = int(width * scale_by)
        height = int(height * scale_by)
        count = image.shape[0]
        return (image, width, height, count,)

    def getsize(self, image):
        width = image.shape[2]
        height = image.shape[1]
        return (width, height)
    
    def getNearestResolution(self, image):
        image_width = image.size()[2]
        image_height = image.size()[1]
        print(f"Input image resolution: {image_width}x{image_height}")
        image_ratio = image_width / image_height
        differences = [
            (abs(image_ratio - resolution[2]), resolution)
            for resolution in self.SDXL_SUPPORTED_RESOLUTIONS
        ]
        smallest = None
        for difference in differences:
            if smallest is None:
                smallest = difference
            else:
                if difference[0] < smallest[0]:
                    smallest = difference
        if smallest is not None:
            width = smallest[1][0]
            height = smallest[1][1]
        else:
            width = 1024
            height = 1024
        print(f"Selected resolution: {width}x{height}")
        return (width, height)