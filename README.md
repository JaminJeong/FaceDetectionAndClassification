## FaceDetectionAndClassification
face detection and classification using tensorflow

## Download model of facenet
```bash 
cd face_recognitions/facenet/src
bash download_and_extract.sh
cd ../model
tree 
.
├── 20180402-114759
│   ├── 20180402-114759.pb
│   ├── model-20180402-114759.ckpt-275.data-00000-of-00001
│   ├── model-20180402-114759.ckpt-275.index
│   └── model-20180402-114759.meta
└── 20180402-114759.zip
```

## Prepare a video and images of face
```bash
cd face_detection/tensorflow-face-detection
mkdir -p media/face_image
cd media
tree
.
├── face_image
│   ├── A.jpg
│   ├── B.jpg
│   ├── C.jpg
│   ├── D.jpg
│   ├── E.jpg
│   └── F.jpg
└── test.mp4

```

## Run
```bash
cd face_detection/tensorflow-face-detection
python inference_video_face.py
```

## Base Implimentation
- https://github.com/yeephycho/tensorflow-face-detection
- https://github.com/davidsandberg/facenet
- https://github.com/velociraptor111/tf-deep-facial-recognition-lite
