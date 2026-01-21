
import os
import json
from sequential_parameter_nodes import Sequential_Parameter_Loader, Sequential_Parameter_Trigger

def test_sequential_logic():
    loader = Sequential_Parameter_Loader()
    trigger = Sequential_Parameter_Trigger()
    
    x_input = "valX1, lblX1\nvalX2, lblX2"
    y_input = "valY1, lblY1\nvalY2, lblY2"
    batch_file = "test_batch.json"
    
    if os.path.exists(batch_file):
        os.remove(batch_file)
        
    print("--- Run 1 ---")
    out1 = loader.load_next_parameters(x_input, y_input, batch_file)
    print(f"Outputs: x={out1[0]}, y={out1[1]}, xl={out1[2]}, yl={out1[3]}")
    trigger.doit("trigger", out1[4])
    
    print("\n--- Run 2 ---")
    out2 = loader.load_next_parameters(x_input, y_input, batch_file)
    print(f"Outputs: x={out2[0]}, y={out2[1]}, xl={out2[2]}, yl={out2[3]}")
    trigger.doit("trigger", out2[4])
    
    print("\n--- Run 3 ---")
    out3 = loader.load_next_parameters(x_input, y_input, batch_file)
    print(f"Outputs: x={out3[0]}, y={out3[1]}, xl={out3[2]}, yl={out3[3]}")
    trigger.doit("trigger", out3[4])
    
    print("\n--- Run 4 ---")
    out4 = loader.load_next_parameters(x_input, y_input, batch_file)
    print(f"Outputs: x={out4[0]}, y={out4[1]}, xl={out4[2]}, yl={out4[3]}")
    trigger.doit("trigger", out4[4])
    
    print("\n--- Run 5 (Should Reset) ---")
    out5 = loader.load_next_parameters(x_input, y_input, batch_file)
    print(f"Outputs: x={out5[0]}, y={out5[1]}, xl={out5[2]}, yl={out5[3]}")

    if os.path.exists(batch_file):
        os.remove(batch_file)

if __name__ == "__main__":
    test_sequential_logic()
