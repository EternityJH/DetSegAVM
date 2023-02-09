# DetSegAVM
There are Detection and Segmentation for this project.

The YOLOv5 and YOLOv8 are the fork repository.

It is recommanded to use the ananconda and create the virtual environment for each model (YOLOv5, YOLOv8, and Segmentation).

## Detection
Please refer to the YOLOv5 and YOLOv8 folder for the environment configuration.

For training and inference with YOLOv5, please modify and run the "YOLOv5_bAVM.ipynb." 

For training and inference with YOLOv8, please modify and run the "YOLOv8_bAVM.ipynb." 

## Segmentation
If run on the anaconda environment please create a virtual environment.

Then run the following code to install proper packages.

The requirements.txt can be found in the Segmentation folder.
```bash
pip install -r requirements.txt  # install
```

For training the segmentation model, please modify and run the "AVM_MR_Seg_train.ipynb."

For inference the segmentation model after training, please modify and run the "AVM_MR_Seg_output.ipynb."

For the combination inference for the detection and segmentation, please modify and run the "AVM_MR_Seg_output_yolo.ipynb."

The configuration detail can be found inside the jupyter notebook.

The code that need to be modified is noted with "#mod."
