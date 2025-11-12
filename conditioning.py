import math

import folder_paths
import comfy.sd
from nodes import ControlNetApplyAdvanced

class conditioning:
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "loader": ("LOADER_PIPE",),
                "prompt": ("PROMPT_PIPE",),
                "enable_controlnet": ("BOOLEAN", {'default': False}),
                "enable_lora": ("BOOLEAN", {'default': False}),
                "apply_lora_trigger_words": ("BOOLEAN", {'default': True}),
                "apply_embedding": ("BOOLEAN", {'default': False}),
                "latent_from": (['latent', 'image'], {'default': 'latent'}),
                "latent_batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional":{
                "controlnet_stack": ("CONTROL_NET_STACK", ),
                "lora_stack": ("LORA_STACK", ),
                "latent": ("LATENT",),
                "image": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ('MODEL', 'CONDITIONING', 'CONDITIONING', 'LATENT', 'VAE', 'CLIP', 'STRING')
    RETURN_NAMES = ('model', 'positive', 'negative', 'latent', 'vae', 'clip', 'help')
 
    FUNCTION = "doit"
 
    #OUTPUT_NODE = False
 
    CATEGORY = "z_mynodes"
 
    def doit(self, loader, prompt, latent_batch_size, enable_controlnet, enable_lora, apply_lora_trigger_words, apply_embedding, latent_from, latent=None, image=None, controlnet_stack=None, lora_stack=None,):

        help = ''
        model, clip, vae = loader
        positive_str, negative_str, embedding_str = prompt

        #apply lora
        if enable_lora and lora_stack is not None:
            
            # Loop through the stack
            triggerWords = []
            for tup in lora_stack:
                defaults = ('none', 1.0, 1.0, [])  # Default values for the tuple
                lora_name, strength_model, strength_clip, lora_trigger_words = (tup + defaults[len(tup):])[:4]
                
                triggerWords.extend(lora_trigger_words)
                lora_path = folder_paths.get_full_path("loras", lora_name)
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                
                model, clip = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)  

            #positive encoding
            if apply_lora_trigger_words:
                positive_str += ', ' + ', '.join(triggerWords)

        tokens = clip.tokenize(positive_str)
        condPos, pooledPos = clip.encode_from_tokens(tokens, return_pooled=True)
        positive_cond = [[condPos, {"pooled_output": pooledPos}]]

        #negative encoding
        if apply_embedding:
            negative_str += ', '.join([embedding_str])

        tokens = clip.tokenize(negative_str)
        condNeg, pooledNeg = clip.encode_from_tokens(tokens, return_pooled=True)
        negative_cond = [[condNeg, {"pooled_output": pooledNeg}]]

        #apply controlnet
        if enable_controlnet and controlnet_stack is not None:
            for controlnet_tuple in controlnet_stack:
                controlnet_name, image_cnet, strength, start_percent, end_percent  = controlnet_tuple
                
                if type(controlnet_name) == str:
                    controlnet_path = folder_paths.get_full_path("controlnet", controlnet_name)
                    controlnet = comfy.controlnet.load_controlnet(controlnet_path)
                else:
                    controlnet = controlnet_name
                
                controlnet_conditioning = ControlNetApplyAdvanced().apply_controlnet(positive_cond, negative_cond,
                                                                                     controlnet, image_cnet, strength,
                                                                                     start_percent, end_percent)

                positive_cond, negative_cond = controlnet_conditioning[0], controlnet_conditioning[1]

        #latent source
        if latent_from == 'image':
            t = vae.encode(image[:,:,:,:3])
            latent = {"samples":t}
        elif latent_from == 'latent':
            pass

        #repeat latent batch
        batch = latent.copy()
        batch_in = latent["samples"]
        
        batch["samples"] = batch_in.repeat((latent_batch_size, 1,1,1))
        if "noise_mask" in latent and latent["noise_mask"].shape[0] > 1:
            masks = latent["noise_mask"]
            if masks.shape[0] < batch_in.shape[0]:
                masks = masks.repeat(math.ceil(batch_in.shape[0] / masks.shape[0]), 1, 1, 1)[:batch_in.shape[0]]
            batch["noise_mask"] = latent["noise_mask"].repeat((latent_batch_size, 1,1,1))
        if "batch_index" in batch:
            offset = max(batch["batch_index"]) - min(batch["batch_index"]) + 1
            batch["batch_index"] = batch["batch_index"] + [x + (i * offset) for i in range(1, latent_batch_size) for x in batch["batch_index"]]

        #return
        help = f"Positive: {positive_str}\nNegative: {negative_str}"

        return (
            model,
            positive_cond, 
            negative_cond, 
            batch,
            vae,
            clip,
            help,
        )
        
        