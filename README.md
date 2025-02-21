# DeepStream-Yolo-Pose-Add-Nms
NVIDIA DeepStream SDK 7.1/ 7.0/ 6.3 / 6.2 / 6.1.1 / 6.1 / 6.0.1 / 6.0 application for YOLO-Pose models
--------------------------------------------------------------------------------------------------
### Improvements on this repository

* Custom ONNX model parser

### Getting started
#### Export model to onnx format with nms
```
from ultralytics import YOLO

# Load a model
# model = YOLO("path/to/best-pose.pt")  # load a custom trained model

# Export the model
model.export(imgsz=640, format="onnx", conf=0.25, iou=0.5, agnostic_nms=False, dynamic=True, nms=True)
```
### Supported models
* [YOLOv11](https://docs.ultralytics.com/vi/tasks/pose)

### Basic usage

#### 1. Download the repo

```
git clone https://github.com/hieptran2k2/DeepStream-Yolo-Pose-Add-Nms.git
cd DeepStream-Yolo-Pose-Add-Nms
```
#### 2. Download the `cfg` and `weights` files from [Ultralytics](https://objects.githubusercontent.com/github-production-release-asset-2e65be/521807533/e1a17d9d-a1cf-4350-82c8-913bfe4da74d?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=releaseassetproduction%2F20250221%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250221T103957Z&X-Amz-Expires=300&X-Amz-Signature=83c0487108b8aaa23bf16cfbe958b791104d6cf918ecf9d90cdd542ab02741b2&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3Dyolo11n-pose.pt&response-content-type=application%2Foctet-stream) repo to the DeepStream-Yolo folder

#### 3. Compile the lib

3.1. Set the `CUDA_VER` according to your DeepStream version

```
export CUDA_VER=XY.Z
```

* x86 platform

  ```
  DeepStream 7.1 = 12.6
  DeepStream 7.0 / 6.4 = 12.2
  DeepStream 6.3 = 12.1
  DeepStream 6.2 = 11.8
  DeepStream 6.1.1 = 11.7
  DeepStream 6.1 = 11.6
  DeepStream 6.0.1 / 6.0 = 11.4
  DeepStream 5.1 = 11.1
  ```

* Jetson platform

  ```
  DeepStream 7.1 = 12.6
  DeepStream 7.0 / 6.4 = 12.2
  DeepStream 6.3 / 6.2 / 6.1.1 / 6.1 = 11.4
  DeepStream 6.0.1 / 6.0 / 5.1 = 10.2
  ```

3.2. Make the lib

```
make -C nvdsinfer_custom_impl_Yolo_pose clean && make -C nvdsinfer_custom_impl_Yolo_pose
```

#### 4. Edit the `config_infer_primary.txt` file according to your model (example for YOLOv4)

```
[property]
...
onnx-file=path/to/model.onnx
model-engine-file=path/to/model.engine
labelfile-path=/path/to/label.txt
...
```
#### 5. Run
```
python main.py -i <uri1> [uri2] -o /path/to/output/file -c /path/to/config/file
```
* Note

|       Flag          |                                   Describe                             |                             Example                          |
| :-----------------: | :--------------------------------------------------------------------: | :----------------------------------------------------------: |
| -i or --input       |      Path to input streams                                             | file:///path/to/file (h264, mp4, ...)  or rtsp://host/video1 |
| -o or --output      |      Path to output file                                               |                          /output/out.mp4                     |
| -c or  --configfile |      Choose the config-file to be used with specified pgie             |                      /model/pgie/config.txt                  |
| --file-loop         |      =Loop the input file sources after EOS if input is file           |                                                              |

**NOTE**: The TensorRT engine file may take a very long time to generate (sometimes more than 10 minutes).

**NOTE**: With DeepStream 7.1, the docker containers do not package libraries necessary for certain multimedia operations like audio data parsing, CPU decode, and CPU encode. This change could affect processing certain video streams/files like mp4 that include audio track. Please run the below script inside the docker images to install additional packages that might be necessary to use all of the DeepStreamSDK features:

```
/opt/nvidia/deepstream/deepstream/user_additional_install.sh
```

### Reference
- DeepStream-Yolo: https://github.com/marcoslucianops/DeepStream-Yolo-Pose
- DeepStream SDK Python: https://github.com/NVIDIA-AI-IOT/deepstream_python_apps
