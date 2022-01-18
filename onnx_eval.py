import onnx
import numpy as np
import onnxruntime as rt
from PIL import Image

# onnx_model = onnx.load("out/yolov2.onnx")

img = Image.open("data/test_images2/face_416.jpg")
img = np.array(img).astype(np.float32)
img = ((img / 255.) - (0.406, 0.456, 0.485)) / (0.225, 0.224, 0.229)
img = np.expand_dims(img, 0).transpose(0, 3, 1, 2).astype(np.float32)
print(img.shape, img.dtype, "max:", img.max(), " min:", img.min())


sess = rt.InferenceSession("out/yolov2.onnx")
input_name = sess.get_inputs()[0].name
pred_onx = sess.run(None, {input_name: img})[0]
print(pred_onx.shape)
print(pred_onx.max(), pred_onx.min())


