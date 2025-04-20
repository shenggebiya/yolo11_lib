
import onnxruntime as ort
# 方法 B：用 onnxruntime
sess = ort.InferenceSession("11.onnx")
print("===== ORT get_inputs =====")
for inp in sess.get_inputs():
    print(inp.name, inp.shape, inp.type)
print("===== ORT get_outputs =====")
for out in sess.get_outputs():
    print(out.name, out.shape, out.type)
