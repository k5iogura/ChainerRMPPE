
# Under construction

[Original REAMDE.md](./README_original.md)

### Environment for test

- python3
- chainer==5.2.0
- ubuntu 16.04

## Download and first move
```
$ git clone https://github.com/k5iogura/Chainer_Realtime_Multi-Person_Pose_Estimation
$ cd Chainer_Realtime_Multi-Person_Pose_Estimation
```

```
$ cd models
$ wget https://raw.githubusercontent.com/ZheC/Realtime_Multi-Person_Pose_Estimation/master/model/_trained_COCO/pose_deploy.prototxt
$ wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
$ wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/face/pose_iter_116000.caffemodel
$ wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel
$ python convert_model.py posenet pose_iter_440000.caffemodel coco_posenet.npz
$ python convert_model.py facenet pose_iter_116000.caffemodel facenet.npz
$ python convert_model.py handnet pose_iter_102000.caffemodel handnet.npz
$ cd ..
```

```
$ python pose_detector.py posenet models/coco_posenet.npz --img data/person.png
$ eog result.png
```

![](files/person_result.png)

and one more,,,

```
$ python pose_detector.py posenet models/coco_posenet.npz --img data/people.png
$ eog result.png
```

![](files/people_result.png)
