import folder_paths
import comfy.controlnet

class controlNetStacker:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        return {
                "required": {
                    "enable": ('BOOLEAN',{'default': True}),
                    "control_net_name": (["None"] + folder_paths.get_filename_list("controlnet"),),
                    "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                },
                "optional": {
                    "control_net_stack": ("CONTROL_NET_STACK",),
                    "image": ("IMAGE",),
                }
        }

    RETURN_TYPES = ("CONTROL_NET_STACK", )
    RETURN_NAMES = ("control_net_stack", )
 
    FUNCTION = "doit"
 
    #OUTPUT_NODE = False
 
    CATEGORY = "z_mynodes"
 
    def doit(self, enable, control_net_name, strength, start_percent, end_percent, image=None, control_net_stack=None):

        if not enable or control_net_name == "None" or image is None:
            return (control_net_stack,)
        
        if enable and control_net_stack is None: #initialize control_net_stack if none is passed in
            control_net_stack = []
        
        #controlnet_path = folder_paths.get_full_path("controlnet", controlnet)
        #controlnet = comfy.controlnet.load_controlnet(controlnet_path)
        control_net_stack.extend([(control_net_name, image, strength, start_percent, end_percent)]),

        return (control_net_stack,)

