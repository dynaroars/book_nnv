import onnx
import base64


def parse_onnx_file(file_path):
    model = onnx.load(file_path)
    return model

def get_string_from_onnx(model):
    return onnx.helper.printable_graph(model.graph)

if __name__ == "__main__":
    model = parse_onnx_file("../data/fnn.onnx")
    serialized_model = get_string_from_onnx(model)
    
    print(model)
    print(serialized_model)