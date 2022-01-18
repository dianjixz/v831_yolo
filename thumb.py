import sys, os, cv2
import numpy as np
import imageio
# from skimage import io
 
def readImg(im_fn):
    im = cv2.imread(im_fn)
    # im = io.imread(im_fn)
    # im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGRA)
    if im is None :
        tmp = imageio.mimread(im_fn)
        if tmp is not None:
            imt = np.array(tmp)
            imt = imt[0]
            im = imt[:,:,0:3]
    if not im is None:
        if im.shape[2] > 3:
            im = im[:, :, :3]
    return im

def thumb_cover(file_path, w, h, out_path=None, force_save=False):
    '''
        @return cv2 image if out_path!=None
    '''
    img = readImg(file_path)
    if type(img) == type(None):
        raise Exception("not support")
    w0 = img.shape[1]
    h0 = img.shape[0]
    r = w/h
    r2 = w0/h0
    if r2 > r:
        cut = int((w0 - r * h0)//2)
        img = img[:, cut:int(w0 - cut), :]
    else:
        cut = int((h0 - w0 * (1/r))//2)
        img = img[cut:int(h0 - cut), :, :]
    img = cv2.resize(img, (w, h))
    if out_path:
        if not force_save and os.path.exists(out_path):
            return False
        dir_ = os.path.dirname(out_path)
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        cv2.imwrite(out_path, img)
        return True
    return img

def thumb_contain(file_path_or_img, w, h, out_path=None, force_save=False, boxes=None, draw_box=False, draw_box_color=(0, 0, 255), draw_box_thickness=2):
    '''
        @box bound box(can be used in YOLO), format: (x1, y1, x2, y2)
        @return cv2 image if out_path!=None
    '''
    if type(file_path_or_img) == np.ndarray:
        img0 = file_path_or_img
    else:
        img0 = readImg(file_path_or_img)
    if type(img0) == type(None):
        raise Exception("not support")
    w0 = img0.shape[1]
    h0 = img0.shape[0]
    r = w/h
    r2 = w0/h0
    img = np.zeros((h, w, 3), dtype=np.uint8)
    if r2 > r:
        new_w = w
        new_h = int(new_w / w0 * h0)
        img0 = cv2.resize(img0, (new_w, new_h))
        half_h = (h - new_h)//2
        img[half_h:(half_h + new_h), :, :] = img0[:, :, :]
        if boxes:
            for i, box in enumerate(boxes):
                box = (int(box[0]*new_w/w0), int(box[1]*new_h/h0+half_h), int(box[2]*new_w/w0), int(box[3]*new_h/h0 + half_h))
                boxes[i] = box
    else:
        new_h = h
        new_w = int(new_h / h0 * w0)
        img0 = cv2.resize(img0, (new_w, new_h))
        half_w = (w - new_w)//2
        img[:, half_w:(half_w + new_w), :] = img0[:, :, :]
        if boxes:
            for i, box in enumerate(boxes):
                box = (int(box[0]*new_w/w0 + half_w), int(box[1]*new_h/h0), int(box[2]*new_w/w0 + half_w), int(box[3]*new_h/h0))
                boxes[i] = box
    if boxes and draw_box:
        for box in boxes:
            cv2.rectangle(img,box[:2],box[2:], draw_box_color, draw_box_thickness)
    if out_path:
        if not force_save and os.path.exists(out_path):
            return False, None
        dir_ = os.path.dirname(out_path)
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        cv2.imwrite(out_path, img)
        return (True, boxes) if boxes else True
    return (img, boxes) if boxes else img


def call(func_name, *args, **kw_args):
    globals()[func_name](*args, **kw_args)

if __name__ == "__main__":
    mode = sys.argv[1]
    file_path = sys.argv[2]
    out_path  = sys.argv[3]
    w = int(sys.argv[4])
    h = int(sys.argv[5])
    
    res = 1
    modes = ["cover", "contain"]
    if mode in modes:
        if os.path.isdir(file_path) and (os.path.isdir(out_path) or not os.path.exists(out_path)):
            is_out_path_dir = os.path.isdir(out_path) or not os.path.exists(out_path)
            for file_name in os.listdir(file_path):
                name, ext = os.path.splitext(file_name)
                if ext == ".jpg" or ext == ".jpeg":
                    if is_out_path_dir:
                        out_file_path = os.path.join(out_path, f"{name}_{w}x{h}.jpg")
                    else:
                        out_file_path = out_path
                    path = os.path.join(file_path, file_name)
                    print("--", out_file_path)
                    call(f"thumb_{mode}", path, w, h, out_file_path, True)
            res = 0
        elif os.path.isfile(file_path) and os.path.exists(file_path):
            if os.path.isdir(out_path):
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                out_path = os.path.join(out_path, f"{file_name}_{w}x{h}.jpg")
            call(f"thumb_{mode}", file_path, w, h, out_path, True)
            res = 0
        else:
            print("path error")
    else:
        print("mode error")
    if res != 0:
        print("error")
    sys.exit(res)
    # img, new_boxes = thumb_contain(file_path, w, h, out_path+"3.jpg", True, boxes=[(50, 10, 270, 230), (160, 30, 230, 100)], draw_box=True)
    # print(new_boxes)

