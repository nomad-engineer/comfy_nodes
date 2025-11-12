import itertools

class prompt_assemble:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "remove_duplicates": ("BOOLEAN", {"default": True})
            },
            "optional": {
                "prompt_a": ('STRING',),
                "prompt_b": ('STRING',),
                "prompt_c": ('STRING',),
                "prompt_d": ('STRING',),
                "prompt_e": ('STRING',),
                "prompt_f": ('STRING',),
                "blacklist": ("STRING",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)

    FUNCTION = "doit"

    CATEGORY = "z_mynodes"

    def doit(self, prompt_a="", prompt_b="", prompt_c="", prompt_d="", prompt_e="", prompt_f="", remove_duplicates=True, blacklist=""):
        # convert strings to lists
        prompt_a_list = list(prompt_a.split(','))
        prompt_b_list = list(prompt_b.split(','))
        prompt_c_list = list(prompt_c.split(','))
        prompt_d_list = list(prompt_d.split(','))
        prompt_e_list = list(prompt_e.split(','))
        prompt_f_list = list(prompt_f.split(','))
        blacklistList = list(blacklist.split(','))

        # concat lists
        promptList = itertools.chain(prompt_a_list,prompt_b_list,prompt_c_list,prompt_d_list,prompt_e_list,prompt_f_list)

        # remove leading and traiing spaces
        promptList = [tag.strip() for tag in promptList]
        blacklistList = [tag.strip() for tag in blacklistList]

        #remove blank tags
        promptList = [tag for tag in promptList if tag not in ["", " "]] 

        #remove blacklisted tags
        promptList = [tag for tag in promptList if tag not in blacklistList]

        # remove duplicates
        if remove_duplicates:
            uniqueTagList = []
            for tag in promptList:
                if tag in uniqueTagList:
                    continue
                else:
                    uniqueTagList.append(tag)
            promptList = uniqueTagList

        # output
        prompt = ', '.join(promptList)
        return (prompt,)

