# ros2 import
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

# address path 
import os
import sys
from pathlib import Path
# add conda env path
sys.path.append(str(Path.home()) + '/anaconda3/envs/torch/lib/python3.10/site-packages')

# yolov5 import
FILE = Path(__file__).resolve()
ROOT = os.path.join(FILE.parents[0], 'yolov5')  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
from ultralytics.utils.plotting import Annotator, colors
from .yolov5.models.common import DetectMultiBackend
from .yolov5.utils.torch_utils import select_device
from .yolov5.utils.general import check_img_size, check_imshow, non_max_suppression, scale_boxes
from .yolov5.utils.augmentations import letterbox

class YoloV5ROS2(Node):
    def __init__(self):
        # ros2 init
        super().__init__('yolov5_ros2')
        self.get_logger().info('yolov5_ros2 node started')
        # ros2 pub/sub
        self.bridge = CvBridge()
        image_topic = '/image_raw'
        image_info_topic = '/camera_info'
        self.image_info_sub = self.create_subscription(CameraInfo, image_info_topic, self.image_info_callback, 10)
        self.imgsz = None
        # wait for first sub of image info
        while self.imgsz is None:
            # logger info throttle
            self.get_logger().info('waiting for image info', throttle_duration_sec=1)
            rclpy.spin_once(self)
        self.destroy_subscription(self.image_info_sub) # we only need to sub camera info once
        self.detection_pub = self.create_publisher(Detection2DArray, '/detections', 10)
        self.detection_result = Detection2DArray()
        # load yolov5 model
        self.half = False
        device = select_device('0') # use gpu 0
        weights = str(ROOT / 'yolov5s.pt')
        data = str(ROOT / 'data/coco128.yaml')
        self.get_logger().info(f'loading yolov5 model with weights in {weights}')
        self.model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        self.get_logger().info(f'yolov5 model warmup')
        self.model.warmup(imgsz=(1, 3, *self.imgsz)) # BCHW
        self.view_img = check_imshow(warn=True) # check if we can view image

        # ros2 sub
        # the callback function depends on yolov5 model, thus init after model
        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)

    def image_info_callback(self, msg:CameraInfo):
        self.imgsz = (msg.height, msg.width) # H, W

    def image_callback(self, msg:Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding= 'bgr8') # HWC
        img0 = img.copy() # for visualization
        # Padded resize
        img = letterbox(img, new_shape=self.imgsz, stride=self.stride, auto=self.pt)[0]
        img = img.transpose(2, 0, 1) # HWC to CHW
        img = np.ascontiguousarray(img) # contiguous array for memory efficiency
        # convert to torch gpu
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if len(img.shape) == 3:
            img = img[None] # expand for batch dim
        # inference
        augment = False # augmented inference
        pred = self.model(img, augment=augment, visualize=False)
        # NMS
        conf_thres = 0.25
        iou_thres = 0.45
        classes = None # optional filter by class
        agnostic_nms = False # class-agnostic NMS
        max_det = 1000 # maximum detections per image
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # process predictions 
        det = pred[0] # we only has one image
        annotator = Annotator(img0, line_width=3, example=str(self.names[0]))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            # self.get_logger().info(f'img shape={img.shape}, img0 shape={img0.shape}', throttle_duration_sec=1)
            # annotate results
            for *xyxy, conf, cls in reversed(det):
                # add to detection result msg
                detection = Detection2D()
                detection.id = self.names[int(cls)]
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                detection.bbox.center.position.x = (x1 + x2) / 2.0
                detection.bbox.center.position.y = (y1 + y2) / 2.0
                detection.bbox.size_x = float(x2 - x1)
                detection.bbox.size_y = float(y2 - y1)
                obj_hypothesis = ObjectHypothesisWithPose()
                obj_hypothesis.hypothesis.class_id = self.names[int(cls)]
                obj_hypothesis.hypothesis.score = float(conf)

                # TODO: add pose info in obj_hypothesis

                detection.results.append(obj_hypothesis)
                self.detection_result.detections.append(detection)

                if self.view_img:
                    c = int(cls)
                    label = f'{self.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
        
        if self.view_img:
            img0 = annotator.result()
            cv2.namedWindow('yolov5_ros2', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO) # allow window resize (Linux)
            cv2.resizeWindow('yolov5_ros2', img0.shape[1], img0.shape[0])
            cv2.imshow('yolov5_ros2', img0)
            cv2.waitKey(1)
        
        self.detection_pub.publish(self.detection_result)
                        

def main(args=None):
    rclpy.init()
    yolov5_node = YoloV5ROS2()
    rclpy.spin(yolov5_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()