import os
import re
import csv
import random
import time
from pathlib import Path
from collections import defaultdict

def count_tags_in_folder(folder_path):
    tag_count = defaultdict(int)
    tag_pattern = re.compile(r'\b[a-zA-Z0-9_]+\b')

    # Iterate through files in folder
    folder_path = Path(folder_path)
    for filename in folder_path.rglob("*.txt"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            content = file.read()
            tags = [tag.strip() for tag in content.split(',')]
            for tag in tags:
                if tag:
                    tag_count[tag] += 1

    # Create CSV output
    csv_output = 'Tag,Count\n'
    for tag, count in sorted(tag_count.items()):
        csv_output += f'{tag},{count}\n'

    return csv_output, tag_count

def generate_random_prompt(tag_count, numberOfTags, weightByFrequency=1.0, blacklist=None):
    if blacklist is None:
        blacklist = []

    print("Blacklisted tags: {}".format(str(blacklist)))

    # Remove blacklisted tags from the tag count
    filtered_tags = {tag: count for tag, count in tag_count.items() if tag not in blacklist}
    #print("Filtered tags: {}".format(str(filtered_tags)))

    # Create weighted pool of tags
    tag_pool = []
    for tag, count in filtered_tags.items():
        no_into_pool = round(count * weightByFrequency)
        if no_into_pool < 1: no_into_pool = 1
        #print(tag)
        tag_pool.extend([tag for i in range(no_into_pool)])

    #for tag in tag_pool:
    #    print(tag)

    # pick tags randomly
    no_of_tags = len(tag_pool)
    prompt = []
    timeout = 1000
    while numberOfTags > 0 and timeout > 0:
        tag = tag_pool[random.randint(0, no_of_tags - 1)]
        #print(tag)
        if tag not in prompt:
            prompt.append(tag)
            numberOfTags = numberOfTags - 1
        timeout = timeout - 1 #stop if there are no more new tags to find

    return ", ".join(prompt)

def clean_tags(tags=[], remove_tags=[], remove_duplicates=True):
    # remove leading and traiing spaces
    
    tags = [tag.strip() for tag in tags]
    
    #remove blank tags
    tags = [tag for tag in tags if tag not in ["", " "]] 

    #remove blacklisted tags
    tags = [tag for tag in tags if tag not in remove_tags]

    # remove duplicates
    if remove_duplicates:
        uniqueTagList = []
        for tag in tags:
            if tag in uniqueTagList:
                continue
            else:
                uniqueTagList.append(tag)
        tags = uniqueTagList
        
    return tags

class prompt_from_tag_files:
    def __init__(self):
        self.last_run_time = 0

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "no_tags": ("INT", {"default": 10, "min": 1, "max": 30, "step": 1}),
                "weight_by_frequency": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "seed": ("INT",)
            },
            "optional": {
                "blacklist": ("STRING",),
                "folder_path": ("STRING",)
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("tag_count", "prompt")

    FUNCTION = "doit"

    CATEGORY = "z_mynodes"

    def doit(self, folder_path, no_tags, weight_by_frequency, blacklist:str, seed):  
        # Add a unique internal trigger to force re-evaluation
        self.last_run_time = time.time()  # This will always update each time the node runs
        unique_trigger = random.random()  # Generate a new random value each time the node runs

        # Use the unique_trigger to force non-deterministic behavior
        csv_output, tag_count = count_tags_in_folder(folder_path)
        blacklist = clean_tags(blacklist.split(','))
        prompt = generate_random_prompt(tag_count, no_tags, weight_by_frequency, blacklist)

        # Combine output with the trigger to ensure uniqueness
        return (csv_output, prompt)

if __name__ == "__main__":
    folder_path = "/home/adam/SanDisk/Files/lora/fat_immobile/fat_immobile_5"
    csv_output, tag_count = count_tags_in_folder(folder_path)
    blacklist = "fatimmobile ,from front"
    blacklist = clean_tags(blacklist.split(','))
    no_tags = 10
    weight_by_frequency = 1
    prompt = generate_random_prompt(tag_count, no_tags, weight_by_frequency, blacklist)
    print(prompt)