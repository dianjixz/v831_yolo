import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import *
import torch.utils.data as data
import numpy as np
import cv2
import tools
import time
import os.path as osp


parser = argparse.ArgumentParser(description='YOLO Detection')
parser.add_argument('-v', '--version', default='yolo_v2',
                    help='yolo_v2, yolo_v3, yolo_v3_spp, slim_yolo_v2, tiny_yolo_v3')
parser.add_argument('-d', '--dataset', default='voc',
                    help='voc, coco-val, custom')
parser.add_argument('-size', '--input_size', default=416, type=int,
                    help='input_size')
parser.add_argument('--trained_model', default='weight/voc/',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--conf_thresh', default=0.1, type=float,
                    help='Confidence threshold')
parser.add_argument('--nms_thresh', default=0.50, type=float,
                    help='NMS threshold')
parser.add_argument('--visual_threshold', default=0.3, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', action='store_true', default=False, 
                    help='use cuda.')
parser.add_argument('--export', action='store_true', default=False, 
                    help='export onnx and ncnn model')

args = parser.parse_args()


def vis(img, bboxes, scores, cls_inds, thresh, class_colors, class_names, class_indexs=None, dataset='voc'):
    if dataset == 'voc' or dataset == "widerface" or dataset == "custom":
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_colors[int(cls_indx)], 1)
                cv2.rectangle(img, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_colors[int(cls_indx)], -1)
                mess = '%s, %.2f' % (class_names[int(cls_indx)], scores[i])
                cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    elif dataset == 'coco-val' and class_indexs is not None:
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_colors[int(cls_indx)], 1)
                cv2.rectangle(img, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_colors[int(cls_indx)], -1)
                cls_id = class_indexs[int(cls_indx)]
                cls_name = class_names[cls_id]
                # mess = '%s: %.3f' % (cls_name, scores[i])
                mess = '%s, %.2f' % (cls_name, scores[i])
                cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    return img
        

def test(net, device, testset, transform, thresh, class_colors=None, class_names=None, class_indexs=None, dataset='voc'):
    num_images = len(testset)
    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        img, _ = testset.pull_image(index)
        h, w, _ = img.shape

        # to tensor
        x = torch.from_numpy(transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
        x = x.unsqueeze(0).to(device)

        t0 = time.time()
        # forward
        bboxes, scores, cls_inds = net(x)
        print("detection time used ", time.time() - t0, "s")
        
        # scale each detection back up to the image
        scale = np.array([[w, h, w, h]])
        # map the boxes to origin image scale
        bboxes *= scale

        img_processed = vis(img, bboxes, scores, cls_inds, thresh, class_colors, class_names, class_indexs, dataset)
        if not os.path.exists("out/test"):
            os.makedirs("out/test")
        cv2.imwrite(f"out/test/{index}.jpg", img_processed)
        # cv2.imshow('detection', img_processed)
        # cv2.waitKey(0)
        # print('Saving the' + str(index) + '-th image ...')
        # cv2.imwrite('test_images/' + args.dataset+ '3/' + str(index).zfill(6) +'.jpg', img)

class TestDatasets(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=['data/test_images2'],
                 transform=None, target_transform=WiderfaceAnnotationTransform(),
                 ):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.ids = list()
        for name in image_sets:
            if os.path.exists(name): # dir
                rootpath = name
                for f_name in os.listdir(name):
                    if f_name.endswith(".jpg") or f_name.endswith(".jpeg"):
                        self.ids.append((osp.join(name, f_name), f_name))
            else: # VOC style data
                rootpath = self.root
                for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                    self.ids.append( (osp.join(rootpath, "JPEGImages", line.strip()), line.strip()) )

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.ids)

    def reset_transform(self, transform):
        self.transform = transform

    def pull_image(self, index):
        path, img_id = self.ids[index]
        return cv2.imread(path, cv2.IMREAD_COLOR), img_id



if __name__ == '__main__':
    # get device
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    input_size = [args.input_size, args.input_size]

    # dataset
    if args.dataset == 'voc':
        print('test on voc ...')
        class_names = VOC_CLASSES
        class_indexs = None
        num_classes = len(VOC_CLASSES)
        dataset = VOCDetection(root=VOC_ROOT, image_sets=[('2007', 'test')], transform=None)
    elif args.dataset == 'widerface':
        print('test on widerface ...')
        class_names = WIDERFACE_CLASSES
        class_indexs = None
        num_classes = 1
        # dataset = WiderfaceDetection(root=WIDERFACE_ROOT, image_sets=['val'], transform=None)
        dataset = TestDatasets(root=WIDERFACE_ROOT)
    elif args.dataset == 'custom':
        print('test on custom ...')
        class_names = CUSTOM_CLASSES
        class_indexs = None
        num_classes = len(CUSTOM_CLASSES)
        dataset = CustomDetection(root=CUSTOM_ROOT, image_sets=['val'], transform=None)
        # dataset = TestDatasets(root=CUSTOM_ROOT)
    elif args.dataset == 'coco-val':
        print('test on coco-val ...')
        class_names = coco_class_labels
        class_indexs = coco_class_index
        num_classes = 80
        dataset = COCODataset(
                    data_dir=coco_root,
                    json_file='instances_val2017.json',
                    name='val2017',
                    img_size=input_size[0])

    class_colors = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(num_classes)]

    # load net
    if args.version == 'yolo_v2':
        from models.yolo_v2 import myYOLOv2
        anchor_size = ANCHOR_SIZE if args.dataset == 'voc' else ANCHOR_SIZE_COCO
        net = myYOLOv2(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, anchor_size=anchor_size)
    
    elif args.version == 'yolo_v3':
        from models.yolo_v3 import myYOLOv3
        anchor_size = MULTI_ANCHOR_SIZE if args.dataset == 'voc' else MULTI_ANCHOR_SIZE_COCO
        net = myYOLOv3(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, anchor_size=anchor_size)
    
    elif args.version == 'yolo_v3_spp':
        from models.yolo_v3_spp import myYOLOv3Spp
        anchor_size = MULTI_ANCHOR_SIZE if args.dataset == 'voc' else MULTI_ANCHOR_SIZE_COCO
        net = myYOLOv3Spp(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, anchor_size=anchor_size)
    
    elif args.version == 'slim_yolo_v2':
        from models.slim_yolo_v2 import SlimYOLOv2 
        anchor_size = ANCHOR_SIZE if args.dataset == 'voc' else (ANCHOR_SIZE_COCO if args.dataset == "coco" else ANCHOR_SIZE_WIDER_FACE)
        net = SlimYOLOv2(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, anchor_size=anchor_size)


    elif args.version == 'tiny_yolo_v3':
        from models.tiny_yolo_v3 import YOLOv3tiny
        anchor_size = TINY_MULTI_ANCHOR_SIZE if args.dataset == 'voc' else TINY_MULTI_ANCHOR_SIZE_COCO
        net = YOLOv3tiny(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, anchor_size=anchor_size)

    net.load_state_dict(torch.load(args.trained_model, map_location=device))
    net.to(device).eval()
    print('Finished loading model!')

    # convert to onnx and ncnn
    from torchsummary import summary
    summary(net.to("cpu"), input_size=(3, input_size[0], input_size[1]), device="cpu")
    if args.export:
        net.no_post_process = True
        from convert import *
        onnx_out="out/yolov2.onnx"
        ncnn_out_param = "out/yolov2.param"
        ncnn_out_bin = "out/yolov2.bin"
        input_shape = (3, input_size[0], input_size[1])
        import os
        if not os.path.exists("out"):
            os.makedirs("out")
        with torch.no_grad():
            torch_to_onnx(net.to("cpu"), input_shape, onnx_out, device="cpu")
            onnx_to_ncnn(input_shape, onnx=onnx_out, ncnn_param=ncnn_out_param, ncnn_bin=ncnn_out_bin)
            print("convert end, ctrl-c to exit")
    net.no_post_process = False

    # evaluation
    with torch.no_grad():
        test(net=net, 
            device=device, 
            testset=dataset,
            transform=BaseTransform(input_size),
            thresh=args.visual_threshold,
            class_colors=class_colors,
            class_names=class_names,
            class_indexs=class_indexs,
            dataset=args.dataset
            )
