# Deep Sort and yolov3_tiny for car detection with PyTorch 
nearly realtime:avg_fps=14 for mx150(2G GPU)

## Latest Update(2020-3-9)
Weights files and demo.mp4 at here:(also i put this video to bilibili,after passing the audit,i will put 链接)
[BaiduDisk]
链接：https://pan.baidu.com/s/1czw0dz56ZjfgrxjBFaUSdg 
提取码：odbu
Yolov3_tiny_1cls_se detection results at here:
链接：https://www.bilibili.com/video/av94365366/

## Dependencies
- python 3 
- numpy
- scipy
- opencv-python
- sklearn
- torch ==1.1.0
- torchvision ==0.3.0
- pillow
- vizer
- edict

## Quick Start
0. Check all dependencies installed

1. Download best.pt(yolo weights) parameters
```
look up
```

3. Download deepsort parameters ckpt.t7
```
look up and put here:
cd deep_sort/deep/checkpoint
```  

Notice:
If compiling failed, the simplist way is to **Upgrade your pytorch >= 1.1 and torchvision >= 0.3" and you can avoid the troublesome compiling problems which are most likely caused by either `gcc version too low` or `libraries missing`.

4. Run my_deep_sort.py
```
my is windows
i do not know linux or ubuntu
my programming ability is very weak
comments are not very good
forgive me
```
Notice:
Maybe some mistakes, you can discuss in issues, help me to modify.
It is just a small work, you can improve based on my code.


## References
- paper: [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)

- code: [nwojke/deep_sort](https://github.com/nwojke/deep_sort)

- paper: [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

- code: [ultralytics/yolov3](https://github.com/ultralytics/yolov3)

- code:[ZQPei/deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch)

