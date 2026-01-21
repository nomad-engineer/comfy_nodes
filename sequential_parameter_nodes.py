import os
import json
from server import PromptServer
from pathlib import Path

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

class Sequential_Parameter_Loader:
    def __init__(self):
        self.current_index = 0
        self.batch_file_path = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("STRING", {"default": '', "multiline": True}),
                "y": ("STRING", {"default": '', "multiline": True}),
                "batch_file_path": ("STRING", {"default": 'temp/batch.json', "multiline": False}),
                "restart_at_batch_end": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (any_type, any_type, "STRING", "STRING", "STRING", "SEQUENTIAL_PARAMETER_TRIGGER",)
    RETURN_NAMES = ("x", "y", "x_label", "y_label", "filename", "to trigger",)
    FUNCTION = "load_next_parameters"
    CATEGORY = "my_nodes"
    OUTPUT_NODE = False

    def _save_state(self, file_path):
        """Save the current index to a JSON file."""
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        state = {
            "current_index": self.current_index,
            "last_filename": getattr(self, "last_filename", "")
        }
        try:
            with open(file_path, "w") as f:
                json.dump(state, f)
        except IOError as e:
            print(f"Error saving state to {file_path}: {e}")

    def _load_state(self, file_path):
        """Load the current index from a JSON file."""
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    content = f.read().strip()
                    if content:
                        state = json.loads(content)
                        self.current_index = state.get("current_index", 0)
                    else:
                        self.current_index = 0
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading state from {file_path}: {e}")
                self.current_index = 0
        else:
            self.current_index = 0

    def parse_input(self, input_str):
        lines = [line.strip() for line in input_str.split('\n') if line.strip()]
        parsed = []
        for line in lines:
            parts = line.split(',', 1)
            value = parts[0].strip()
            label = parts[1].strip() if len(parts) > 1 else value
            parsed.append((value, label))
        return parsed

    def load_next_parameters(self, x, y, batch_file_path, restart_at_batch_end):
        self.batch_file_path = batch_file_path
        self.restart_at_batch_end = restart_at_batch_end
        self._load_state(batch_file_path)

        x_list = self.parse_input(x)
        y_list = self.parse_input(y)

        if not x_list: x_list = [("", "")]
        if not y_list: y_list = [("", "")]

        total_combinations = len(x_list) * len(y_list)
        
        if self.current_index >= total_combinations:
            if restart_at_batch_end:
                self.current_index = 0
                self._save_state(batch_file_path)
            else:
                # We stayed at the end, next run will also hit this if not restarted
                raise ValueError("Batch Complete")

        idx_x = self.current_index // len(y_list)
        idx_y = self.current_index % len(y_list)

        val_x, lbl_x = x_list[idx_x]
        val_y, lbl_y = y_list[idx_y]

        # Convert numeric values if possible
        def try_convert(v):
            try:
                if '.' in v: return float(v)
                return int(v)
            except:
                return v

        val_x = try_convert(val_x)
        val_y = try_convert(val_y)

        self.total_combinations = total_combinations

        # Generate combined filename from labels
        labels_to_join = [l for l in [lbl_x, lbl_y] if l]
        filename = "__".join(labels_to_join)
        self.last_filename = filename

        return (val_x, val_y, lbl_x, lbl_y, filename, self,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always change to ensure it runs when triggered
        import random
        return (float("NaN"),)

class Sequential_Parameter_Trigger:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "trigger": (any_type,),
                    "from_loader": ("SEQUENTIAL_PARAMETER_TRIGGER",),
                    "auto_submit_workflow": ("BOOLEAN", {"default": True}),
                    },
                "optional": {
                    "filename": ("STRING", {"forceInput": True}),
                }
            }

    FUNCTION = "doit"
    CATEGORY = "my_nodes"
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("trigger_pass_through",)
    OUTPUT_NODE = True

    def doit(self, trigger, from_loader, auto_submit_workflow, filename=None):
        loader = from_loader
        loader.current_index += 1
        
        # Determine filename to save: use provided one if not empty, otherwise from loader
        final_filename = filename if filename else getattr(loader, "last_filename", "")
        loader.last_filename = final_filename
        
        print(f"Sequential Trigger: Index {loader.current_index}/{loader.total_combinations}, Filename: {final_filename}, Auto-Submit: {auto_submit_workflow}")

        if loader.current_index < loader.total_combinations:
            loader._save_state(loader.batch_file_path)
            if auto_submit_workflow:
                print("Sequential Trigger: Sending auto-submit signal")
                PromptServer.instance.send_sync("impact-add-queue", {})
                PromptServer.instance.send_sync("execution_start", {"node_id": "sequential_parameter_trigger"}) 
        else:
            # We reached the end of the batch
            if loader.restart_at_batch_end:
                loader.current_index = 0
                loader._save_state(loader.batch_file_path)
                print("Sequential Parameter Sweep Finished - Re-starting from beginning")
                if auto_submit_workflow:
                    print("Sequential Trigger: Sending auto-submit signal for restart")
                    PromptServer.instance.send_sync("impact-add-queue", {})
                    PromptServer.instance.send_sync("execution_start", {"node_id": "sequential_parameter_trigger"})
            else:
                # Stay at the end index so the loader can "prompt" on next run
                loader._save_state(loader.batch_file_path)
                print("Sequential Parameter Sweep Finished")

        return (trigger,)

class Lora_List:
    @classmethod
    def INPUT_TYPES(s):
        import folder_paths
        LORA_LIST = ["None"] + sorted(folder_paths.get_filename_list("loras"), key=str.lower)
        return {
            "required": {
                "lora_1": (LORA_LIST,),
                "lora_2": (LORA_LIST,),
                "lora_3": (LORA_LIST,),
                "lora_4": (LORA_LIST,),
                "lora_5": (LORA_LIST,),
                "lora_6": (LORA_LIST,),
                "lora_7": (LORA_LIST,),
                "lora_8": (LORA_LIST,),
                "lora_9": (LORA_LIST,),
                "lora_10": (LORA_LIST,),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_list",)
    FUNCTION = "generate_list"
    CATEGORY = "my_nodes"

    def generate_list(self, **kwargs):
        loras = []
        for i in range(1, 11):
            val = kwargs.get(f"lora_{i}")
            if val and val != "None":
                loras.append(val)
        return ("\n".join(loras),)

class Lora_List_From_Path:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_path": ("STRING", {"default": '', "multiline": False}),
                "recursion_depth": ("INT", {"default": 0, "min": 0, "max": 10}),
                "include": ("STRING", {"default": '*', "multiline": False}),
                "exclude": ("STRING", {"default": '', "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_list",)
    FUNCTION = "generate_list"
    CATEGORY = "my_nodes"

    def generate_list(self, base_path, recursion_depth, include, exclude):
        import re
        import folder_paths
        
        # Get the standard ComfyUI lora list (paths relative to lora folder)
        all_loras = folder_paths.get_filename_list("loras")
        
        print(f"Lora List From Path: Filtering {len(all_loras)} total loras for path: '{base_path}', inc: '{include}', exc: '{exclude}'")

        loras = []
        
        def get_regex_from_comma_string(s):
            if not s or s == "*":
                return None
            # Split by comma, strip whitespace, and create an OR regex
            parts = [re.escape(p.strip()) for p in s.split(",") if p.strip()]
            if not parts:
                return None
            return re.compile("|".join(parts), re.IGNORECASE)

        # Regex search patterns
        try:
            include_re = get_regex_from_comma_string(include)
            exclude_re = get_regex_from_comma_string(exclude)
        except re.error as e:
            print(f"Lora List From Path: Invalid regex construction: {e}")
            return ("",)

        found_count = 0
        include_count = 0
        exclude_count = 0

        for rel_file in all_loras:
            # First, check if the lora is within the specified 'base_path' (substring match)
            if base_path and base_path not in rel_file:
                continue
            
            # Depth check
            path_to_check = rel_file
            if base_path:
                parts = rel_file.split(base_path, 1)
                path_to_check = parts[1].lstrip('/\\')
            
            current_depth = len(Path(path_to_check).parent.parts) if path_to_check else 0
            if current_depth > recursion_depth:
                continue

            found_count += 1
            
            # Check include (if include is empty or *, it matches all)
            passes_include = True
            if include_re:
                passes_include = bool(include_re.search(rel_file))
            
            if passes_include:
                include_count += 1
                # Check exclude
                is_excluded = False
                if exclude_re:
                    is_excluded = bool(exclude_re.search(rel_file))
                
                if not is_excluded:
                    loras.append(rel_file)
                else:
                    exclude_count += 1

        if len(loras) > 0:
            print(f"--- Found {len(loras)} Lora Paths ---")
            
            # Generate labels
            labels = []
            for l in loras:
                # Remove extension
                label = os.path.splitext(l)[0]
                # Replace '/' with '_'
                label = label.replace('/', '_').replace('\\', '_')
                labels.append(label)
            
            # Find and remove common prefix
            if len(labels) > 1:
                # Find common prefix using os.path.commonprefix on the parts
                # Actually, simpler to check strings if we are consistent
                common = os.path.commonprefix(labels)
                if common:
                    # Only remove if it ends with a separator-turned-underscore for safety
                    # Or just remove what is common to keep it short as requested
                    labels = [l[len(common):] if l.startswith(common) else l for l in labels]
            
            # Create the multiline output with labels
            lines = []
            for i in range(len(loras)):
                lines.append(f"{loras[i]}, {labels[i]}")
            
            for l in lines:
                print(f"  - {l}")
            print("------------------------")
            return ("\n".join(lines),)
        else:
            print("Lora List From Path: NO LORAS FOUND MATCHING CRITERIA")
            return ("",)
