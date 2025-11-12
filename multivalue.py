class ComplexValueNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "complex_value": ("COMPLEX", {}),
            }
        }

    RETURN_TYPES = ("COMPLEX",)
    RETURN_NAMES = ("complex_value_out",)
    FUNCTION = "process_values"
    CATEGORY = "Custom"

    def process_values(self, complex_value):
        # Process the values as needed
        processed_values = complex_value  # Example processing
        return (processed_values,)

NODE_CLASS_MAPPINGS = {"ComplexValueNode": ComplexValueNode}
NODE_DISPLAY_NAME_MAPPINGS = {"ComplexValueNode": "Complex Value Node"}

