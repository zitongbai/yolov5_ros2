# yolov5_ros2
use yolov5 in ros2 humble

# Usage
download this repo
```shell
git clone git@github.com:zitongbai/yolov5_ros2.git --recurse-submodules
```

then build the ros2 workspace

make sure you have ROS2 package `usb_cam` and `vision msgs`, for example: 
```shell
sudo apt update
sudo apt install ros-humble-usb-cam
sudo apt install ros-humble-vision-msgs
```
in one terminal, run usb cam:
```shell
ros2 run usb_cam usb_cam_node_exe
```

in another terminal, run the python node:
```shell
ros2 run yolov5_ros2 detect_cam
```

you would get the image annotated

<img src="./detect%20result.png" width = "500" alt="detect"/>

# Acknowledge

* [fishros/yolov5_ros2](https://github.com/fishros/yolov5_ros2)
* [yolov5](https://github.com/ultralytics/yolov5)
