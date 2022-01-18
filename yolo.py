from maix import nn
from PIL import Image, ImageDraw, ImageFont
from maix import  display
import time
from maix.nn import decoder
from maix import camera


class funation:
    model = {
        "param": "/root/yolov2_int8.param",
        "bin": "/root/yolov2_int8.bin"
    }
    options = {
        "model_type":  "awnn",
        "inputs": {
            "input0": (224, 224, 3)
        },
        "outputs": {
            "output0": (7, 7, (1+4+3)*5)
        },
        "mean": [127.5, 127.5, 127.5],
        "norm": [0.0078125, 0.0078125, 0.0078125],
    }
    m = None
    yolo2_decoder = None
    w = 224
    h = 224
    labels = ["mouse","sipeed_logo"]
    anchors = [1.19, 1.98, 2.79, 4.59, 4.53, 8.92, 8.06, 5.29, 10.32, 10.65]
    def __init__(self):
        camera.config(size=(224, 224))
        self.m = nn.load(self.model, opt=self.options)
        self.yolo2_decoder = decoder.Yolo2(len(self.labels), self.anchors, net_in_size=(self.w, self.h), net_out_size=(7, 7))
    def draw_rectangle_with_title(self,img, box, disp_str, bg_color=(255, 0, 0), font_color=(255, 255, 255)):
        font = ImageFont.load_default()
        font_w, font_h = font.getsize(disp_str)
        img.rectangle((box[0], box[1], box[0] + box[2], box[1] + box[3]), fill=None, outline=bg_color, width=2)
        img.rectangle((box[0], box[1] - font_h, box[0] + font_w, box[1]), fill=bg_color)
        img.text((box[0], box[1] - font_h), disp_str, fill=font_color, font=font)
    def run(self):
        img = camera.read()
        t = time.time()
        out = self.m.forward(img, quantize=True, layout="hwc")
        # print(out)
        print("-- forward: ", time.time() - t )
        t = time.time()
        boxes, probs = self.yolo2_decoder.run(out, nms=0.3, threshold=0.3, img_size=(self.w, self.h))
        print("-- decode: ", time.time() - t )
        print(len(boxes))
        draw = display.get_draw()
        if len(boxes):
            for i, box in enumerate(boxes):
                class_id = probs[i][0]
                prob = probs[i][1][class_id]
                disp_str = "{}:{:.2f}%".format(self.labels[class_id], prob*100)
                self.draw_rectangle_with_title(draw, box, disp_str)
            display.show()
        else:
            display.show()


if __name__ == "__main__":
    import signal
    def handle_signal_z(signum,frame):
        print("erzi over")
        exit(0)
    signal.signal(signal.SIGINT,handle_signal_z)
    start = funation()
    while True:
        start.run()



