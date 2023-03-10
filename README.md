# Object-Detection-Tracking
This repository investigates the performance of three different object trackers (Basic Iou Assignment, Hungarian bipartite matching, and ReID tracking) on the MOT16 datasets.
The [Faster R-CNN object detector](https://arxiv.org/abs/1506.01497) is used to provide detections for the trackers.

This work was part of the Masters course "Computer Vision 3 : Detection, Segmentation & Tracking" at the Technical University of Munich.


## Datasets
- [MOT16](https://motchallenge.net/data/MOT16/)
- [Market-1501](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)

## Getting Started
### Installation
1. Clone this repository
 <br>`git clone https://github.com/MahmouddAhmed/Object-Detection-Tracking.git`<br>

2. Create a conda environment from the environment.yml file<br>`conda env create -f environment.yml`<br>

3. Activate the environment<br>`conda activate CV3`<br>
4. Create a `data` directory to store the MOT16 and Market-1501 datasets.<br>`mkdir data`<br>
5. Download the MOT16 and Market-1501 datasets and extract them to the `data` directory.
6. Create a `models` directory to store the trained models.<br>`mkdir models`<br>
7. You can Download the pre-trained models from [ReID Network](https://drive.google.com/file/d/1nn_q3EiysGlk5h13xdnDyE167IdJCySz/view?usp=share_link) and [Faster-RCNN Detector](https://drive.google.com/file/d/16XTesdELSJ-xRaE9E70JgzMI0vD1a943/view?usp=share_link) and add them to the `models` directory.



### Usage

Please make sure to place the data and models in the correct folder before running the script otherwise the program will give error.

* Run the training script to train a siamese Reid Network on the Market1501 Dataset

    ```
    python src/train.py [OPTIONS]
    ```
#### Options:

|Option|Descreption|Default Value|      
|------|-----------|-------------| 
|batch_size |batch size during training|32|
|cross_entropy_weight|weight for cross entropy loss|1.0|
|data_dir |Market 1501 data directory|./| 
|epoch |number of epoch to run|50|
|epoch_eval_freq |evaluate model every n epochs|5|
|gpu |use gpu if available|0|
|hard_negative_mining_weight|weight of hard negative mining loss|1.0|
|learning_rate |Initial learning rate |0.0003|
|loss_margin |margin for triplet loss (Margin ranking loss) |0.1|
|model |name of the stored model file|final-reid_model.pth|
|output_dir |The directory which the model will be saved|./models|
|print_freq |print losses every n steps|10|
|resume_ckpt |checkpoint path to resume training from|None|

* Run the a tracker on a MOT16 sequence

    ```
    python src/run_tracker.py [OPTIONS]
    ```
#### Options:

|Option|Descreption|Default Value|      
|------|-----------|-------------| 
|data_dir |MOT16 data directory|data/MOT16| 
|gpu |use gpu if available|0|
|nms_thresh |non-maximum suppression applied on object detections|0.3|
|obj_detect_path |Path to the object detector model|models/faster_rcnn_fpn.model|
|output_dir |output directory to save the results|outputs/results/|
|patience |The number of frames to keep an unmatched track|60|
|reid_network_path |Path of the trained reid network|models/final-reid_model.pth|
|seq_name|The sequences to run the tracker on|MOT16-train|
|tracker |An integer That represents the tracker type to use 1: IOU Assignment \n 2: Hungarian IOU Tracker \n 3: ReID Tracker|1|

* You can also check the tracker.ipynb notebook inside notebook directory. To view the visualization of the MOT16 Datasets and the visualization results of the object detector in additon to the visualizations for each tracker

## Results
The results of the run_tracker will be printed to the console and saved in the `outputs` directory.

## Acknowledgments
- The authors of the MOT16 and Market-1501 datasets for making their data publicly available.
- The authors of the Faster R-CNN object detector for their work on object detection.
- The Computer Vision 3 : Detection, Segmentation & Tracking course at the Technical University of Munich for providing the opportunity to work on this project.
- The authors of the papers [1], [2],[3],[4] and [5] for their research on object tracking that served as inspiration for this project.

## References
[1] A. Milan, L. Leal-Taixé, I. Reid, S. Roth, and K. Schindler, “MOT16: A Benchmark for Multi-Object Tracking,” arXiv:1603.00831 [cs], 2016.

[2] L. Zheng, L. Shen, L. Tian, S. Wang, J. Wang, and Q. Tian, “Scalable Person Re-identification: A Benchmark,” in Proceedings of the IEEE International Conference on Computer Vision, 2015.

[3] S. Ren, K. He, R. Girshick and J. Sun, “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,” CoRR, vol. abs/1506.01497, 2015. [Online]. Available: http://arxiv.org/abs/1506.01497. 

[4] F. Schroff, D. Kalenichenko and J. Philbin, “FaceNet: A Unified Embedding for Face Recognition and Clustering,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2015.

[5] M. Hermans, B. Beymer, P. B. N. G. Van Rooyen and A. W. Smeulders, “In Defense of the Triplet Loss for Person Re-Identification,” arXiv:1703.07

