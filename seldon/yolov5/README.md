# Yolo V5
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-79FFE1)](https://pytorch.org)
![Computer Vision](https://img.shields.io/badge/Type-Computer%20Vision-79FFE1)

One of the most popular and most powerful computer vision algorithms in the field of real-time object detection. V5 version outperforms all the previous versions and got near to EfficientDet AP with higher FPS.


## Deploy 
Click a button to deploy a model with [Syndicai](https://syndicai.co).

[![Syndicai-Deploy](https://raw.githubusercontent.com/syndicai/brand/main/button/deploy.svg)](https://app.syndicai.co/newModel?repository=https://github.com/syndicai/models/tree/master/seldon/yolov5)


## Example
| input | output |
| --- | --- |
| <img src="sample_data/input.jpeg" width="410"> | <img src="sample_data/output.png" width="410"> |


## Run Locally
Execute following commands in order to run a model locally.
```bash
# Download and run docker container
docker run -v ${PWD}:/mnt/workspace:ro -p 8000:8000 syndicai/engine:python3.7 local

# Run Model
curl -X POST http://localhost:8000/predict \
  --header 'Content-Type: application/json' \
  --data '{
	"strData": "https://images.pexels.com/photos/2083866/pexels-photo-2083866.jpeg?auto=compress&cs=tinysrgb&h=750&w=1260"
}'
```

## Reference
Ultralytics [yolov5](https://github.com/ultralytics/yolov5) open-source research into future object detection methods.