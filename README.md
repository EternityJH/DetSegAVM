# DetSegAVM
There are Detection and Segmentation for this project.
The YOLOv5 and YOLOv8 are the fork repository.

## Detection


## Segmentation
If run on the anaconda environment please create a virtual environment.

Then run the following code to install proper packages.

The requirements.txt can be found in the Segmentation folder.
```bash
pip install -r requirements.txt  # install
```

For training the segmentation model, please run the "AVM_MR_Seg_train.ipynb."

For inference the segmentation model after training, please run the "AVM_MR_Seg_output.ipynb."

For the combination inference for the detection and segmentation, please run the "AVM_MR_Seg_output_yolo.ipynb."

The configuration detail can be found inside the jupyter notebook.
