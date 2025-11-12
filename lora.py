import folder_paths
import hashlib
import json
import os
import requests
import shutil

def get_preview_path(name, type):
    file_name = os.path.splitext(name)[0]
    file_path = folder_paths.get_full_path(type, name)

    if file_path is None:
        print(f"Unable to get path for {type} {name}")
        return None
    
    file_path_no_ext = os.path.splitext(file_path)[0]
    item_image=None
    for ext in ["png", "jpg", "jpeg", "preview.png"]:
        has_image = os.path.isfile(file_path_no_ext + "." + ext)
        if has_image:
            item_image = f"{file_name}.{ext}"
            break
        
    return has_image, item_image


def copy_preview_to_temp(file_name):
    if file_name is None:
        return None, None
    base_name = os.path.basename(file_name)
    lora_less = "/".join(file_name.split("/")[1:])

    file_path = folder_paths.get_full_path("loras", lora_less)
    if file_path is None:
        return None, None

    temp_path = folder_paths.get_temp_directory()
    preview_path = os.path.join(temp_path, "lora_preview")
    if not os.path.isdir(preview_path) :
        os.makedirs(preview_path)
    preview_path = os.path.join(preview_path, base_name)


    shutil.copyfile(file_path, preview_path)
    return preview_path, base_name

# add previews in selectors
def populate_items(names, type):
    for idx, item_name in enumerate(names):

        has_image, item_image = get_preview_path(item_name, type)

        names[idx] = {
            "content": item_name,
            "image": f"{type}/{item_image}" if has_image else None,
            "type": "loras",
        }
    names.sort(key=lambda i: i["content"].lower())


def load_json_from_file(file_path):
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")
        raise

def save_dict_to_json(data_dict, file_path):
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data_dict, json_file, indent=4)
            print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving JSON to file: {e}")

def get_model_version_info(hash_value):
    api_url = f"https://civitai.com/api/v1/model-versions/by-hash/{hash_value}"
    response = requests.get(api_url)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None
    
def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def load_and_save_tags(lora_name, force_fetch):
    json_tags_path = "./loras_tags.json"
    lora_tags = load_json_from_file(json_tags_path)
    output_tags = lora_tags.get(lora_name, None) if lora_tags is not None else None
    if output_tags is not None:
        output_tags_list = output_tags
    else:
        output_tags_list = []

    lora_path = folder_paths.get_full_path("loras", lora_name)
    if lora_tags is None or force_fetch or output_tags is None: # search on civitai only if no local cache or forced
        print("[Lora-Auto-Trigger] calculating lora hash")
        LORAsha256 = calculate_sha256(lora_path)
        print("[Lora-Auto-Trigger] requesting infos")
        model_info = get_model_version_info(LORAsha256)
        if model_info is not None:
            if "trainedWords" in model_info:
                print("[Lora-Auto-Trigger] tags found!")
                if lora_tags is None:
                    lora_tags = {}
                lora_tags[lora_name] = model_info["trainedWords"]
                save_dict_to_json(lora_tags, json_tags_path)
                output_tags_list = model_info["trainedWords"]
        else:
            print("[Lora-Auto-Trigger] No informations found.")
            if lora_tags is None:
                    lora_tags = {}
            lora_tags[lora_name] = []
            save_dict_to_json(lora_tags,json_tags_path)

    return output_tags_list

def show_list(list_input):
    i = 0
    output = ""
    for debug in list_input:
        output += f"{i} : {debug}\n"
        i+=1
    return output

def get_metadata(filepath, type):
    filepath = folder_paths.get_full_path(type, filepath)
    with open(filepath, "rb") as file:
        # https://github.com/huggingface/safetensors#format
        # 8 bytes: N, an unsigned little-endian 64-bit integer, containing the size of the header
        header_size = int.from_bytes(file.read(8), "little", signed=False)

        if header_size <= 0:
            raise BufferError("Invalid header size")

        header = file.read(header_size)
        if header_size <= 0:
            raise BufferError("Invalid header")
        header_json = json.loads(header)
        return header_json["__metadata__"] if "__metadata__" in header_json else None
    
# parse the __metadata__ json looking for trained tags 
def sort_tags_by_frequency(meta_tags):
    if meta_tags is None:
        return []
    if "ss_tag_frequency" in meta_tags:
        meta_tags = meta_tags["ss_tag_frequency"]
        meta_tags = json.loads(meta_tags)
        sorted_tags = {}
        for _, dataset in meta_tags.items():
            for tag, count in dataset.items():
                tag = str(tag).strip()
                if tag in sorted_tags:
                    sorted_tags[tag] = sorted_tags[tag] + count
                else:
                    sorted_tags[tag] = count
        # sort tags by training frequency. Most seen tags firsts
        sorted_tags = dict(sorted(sorted_tags.items(), key=lambda item: item[1], reverse=True))
        return list(sorted_tags.keys())
    else:
        return []

def parse_selector(selector, tags_list): 
    if len(tags_list) == 0:
        return ""
    range_index_list = selector.split(",")
    output = {}
    for range_index in range_index_list:
        # single value
        if range_index.count(":") == 0:
            # remove empty values
            if range_index.strip() == "":
                continue
            index = int(range_index)
            # ignore out of bound indexes
            if abs(index) > len(tags_list) - 1:
                continue
            output[index] = tags_list[index]

        # actual range
        if range_index.count(":") == 1:
            indexes = range_index.split(":")
            # check empty
            if indexes[0] == "":
                start = 0
            else:
                start = int(indexes[0])
            if indexes[1] == "":
                end = len(tags_list)
            else:
                end = int(indexes[1])
            # check negative
            if start < 0:
                start = len(tags_list) + start
            if end < 0:
                end = len(tags_list) + end
            # clamp start and end values within list boundaries
            start, end = min(start, len(tags_list)), min(end, len(tags_list))
            start, end = max(start, 0), max(end, 0)
            # merge all
            for i in range(start, end):
                output[i] = tags_list[i]
    return ", ".join(list(output.values()))

def append_lora_name_if_empty(tags_list, lora_path, enabled):
    if not enabled or len(tags_list) > 0:
        return tags_list
    filename = os.path.splitext(lora_path)[0]
    filename = os.path.basename(filename)

    tags_list.append(filename)
    return tags_list


#------------------------------------------------

class LoraStackerTrigger:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        LORA_LIST = sorted(folder_paths.get_filename_list("loras"), key=str.lower)
        return {
            "required": { 
                "enable": ("BOOLEAN", {"default": True}),
                "lora_name": (LORA_LIST, ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "force_fetch": ("BOOLEAN", {"default": False}),
                "append_loraname_if_empty": ("BOOLEAN", {"default": False}),
                "add_trigger_words_to_stack": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "lora_stack": ("LORA_STACK", ),
                "override_lora_name":("STRING", {"forceInput": True}), 
            }
        }
    
    RETURN_TYPES = ("LORA_STACK", "STRING", "STRING",)
    RETURN_NAMES = ("lora_stack", "lora_trigger_words", "meta_tags_list",)
    FUNCTION = "doit"
    CATEGORY = "z_mynodes"

    def doit(self, enable, lora_name, strength_model, strength_clip, force_fetch, append_loraname_if_empty, add_trigger_words_to_stack, lora_stack=None, override_lora_name=""):
        if not enable:
            return (lora_stack, '', '',)
        
        if override_lora_name != "":
            lora_name = override_lora_name
        meta_tags_list = sort_tags_by_frequency(get_metadata(lora_name, "loras"))
        lora_trigger_words = load_and_save_tags(lora_name, force_fetch)

        meta_tags_list = append_lora_name_if_empty(meta_tags_list, lora_name, append_loraname_if_empty)
        lora_trigger_words = append_lora_name_if_empty(lora_trigger_words, lora_name, append_loraname_if_empty)

        if lora_stack is None:
            lora_stack = []

        if add_trigger_words_to_stack:
            lora_stack.extend([(lora_name, strength_model, strength_clip, lora_trigger_words,)])
        else:
            lora_stack.extend([(lora_name, strength_model, strength_clip,)])
  
        return (lora_stack, lora_trigger_words, meta_tags_list,)
    
class LoraStackGetTriggerWords:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": { 
                "lora_stack": ("LORA_STACK", ),
                "force_fetch": ("BOOLEAN", {"default": False}),
                "append_loraname_if_empty": ("BOOLEAN", {"default": False}),
                "add_trigger_words_to_stack": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "lora_stack": ("LORA_STACK", )
            }
        }
    
    RETURN_TYPES = ("LORA_STACK", "STRING", "STRING",)
    RETURN_NAMES = ("lora_stack", "lora_trigger_words", "meta_tags_list",)
    FUNCTION = "doit"
    CATEGORY = "z_mynodes"

    def doit(self, force_fetch, append_loraname_if_empty, lora_stack, add_trigger_words_to_stack):

        triggerWords = ''
        metaWords = ''
        lora_stack_out = []

        for tup in lora_stack:
            defaults = ('none', 1.0, 1.0, [])  # Default values for the tuple
            lora_name, strength_model, strength_clip, lora_trigger_words = (tup + defaults[len(tup):])[:4]

            meta_tags_list = sort_tags_by_frequency(get_metadata(lora_name, "loras"))
            lora_trigger_words = load_and_save_tags(lora_name, force_fetch)

            meta_tags_list = append_lora_name_if_empty(meta_tags_list, lora_name, append_loraname_if_empty)
            lora_trigger_words = append_lora_name_if_empty(lora_trigger_words, lora_name, append_loraname_if_empty)

            triggerWords += ', ' + ', '.join(lora_trigger_words)
            metaWords +=  ', ' + ', '.join(meta_tags_list)

            if add_trigger_words_to_stack:
                lora_stack_out.extend([(lora_name, strength_model, strength_clip, lora_trigger_words,)])
            else:
                lora_stack_out.extend([(lora_name, strength_model, strength_clip,)])
  

        return (lora_stack_out, triggerWords, metaWords,)