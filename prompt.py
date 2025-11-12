class promptPipeOutput:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("PROMPT_PIPE", )
            },
            "optional": {
            },
        }
 
    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("positive", "negative", "embedding",)
 
    FUNCTION = "doit"
 
    #OUTPUT_NODE = False
 
    CATEGORY = "z_mynodes"
 
    def doit(self, prompt):  
        positive, negative, emmbedding = prompt

        return (positive, negative, emmbedding,)


class promptSimple:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "positive": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "negative": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "embedding": ("STRING", {"multiline": True, "dynamicPrompts": True})       
            },
        }
 
    RETURN_TYPES = ("PROMPT_PIPE",)
    RETURN_NAMES = ("prompt", )
 
    FUNCTION = "doit"
 
    #OUTPUT_NODE = False
 
    CATEGORY = "z_mynodes"
 
    def doit(self, positive='', negative='', embedding=''):  
        pipe_line = positive, negative, embedding

        return (pipe_line, )
    
class promptPony:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "positive": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "negative": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "embedding": ("STRING", {"multiline": True, "dynamicPrompts": True})       
            },
        }
 
    RETURN_TYPES = ("PROMPT_PIPE",)
    RETURN_NAMES = ("prompt", )
 
    FUNCTION = "doit"
 
    #OUTPUT_NODE = False
 
    CATEGORY = "z_mynodes"
 
    def doit(self, positive='', negative='', embedding=''):  
        pipe_line = positive, negative, embedding

        return (pipe_line, )