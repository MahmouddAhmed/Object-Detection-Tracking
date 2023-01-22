import os
import time
import torch
import argparse
from utils.utils import run_tracker
from data.mot_16_track import MOT16Sequences
from models.object_detector import FRCNN_FPN
from trackers.iou_tracker import IoUAssignmentTracker
from trackers.reid_hungarian_iou_tracker import ReIDTracker
from trackers.hungarian_iou_tracker import HungarianIoUTracker

def parse_args():
    parser = argparse.ArgumentParser('Tracker')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='0',help='GPU', dest='gpu')
    parser.add_argument('--nms_thresh', type=float, default=0.3, help='non-maximum suppression applied on object detections')
    parser.add_argument('--obj_detect_path', type=str,default='models/faster_rcnn_fpn.model', help='Path to the object detector model')
    parser.add_argument('--reid_network_path', type=str,default='models/final-reid_model.pth', help='Path to the reid feature extractor model')
    parser.add_argument('--seq_name', type=str, default='MOT16-train', help='The sequences to run the tracker on')
    parser.add_argument('--output_dir', type=str, default='outputs/results/', help='output directory to save the results')
    parser.add_argument('--patience', type=int, default=60, help='The number of frames to keep an unmatched track')
    parser.add_argument('--data_dir', default='data/MOT16', type=str, help='The directory that contains the MOT-16 Sequences')
    parser.add_argument('--tracker', default=1, type=int, help='An integer That represents the tracker type to use 1: IOU Assignment \n 2: Hungarian IOU Tracker \n 3: ReID Tracker')
    


    return parser.parse_args()

def main(args):

   


    # Declare device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    # object detector
    print(args.obj_detect_path)
    obj_detect_nms_thresh = args.nms_thresh
    obj_detect = FRCNN_FPN(num_classes=2, nms_thresh=obj_detect_nms_thresh)
    obj_detect_state_dict = torch.load(args.obj_detect_path,map_location=lambda storage, loc: storage)
    obj_detect.load_state_dict(obj_detect_state_dict)
    obj_detect.eval()
    obj_detect.to(device)

    
    
    seq_name = args.seq_name
    output_dir=args.output_dir
    inactive_pateince=args.patience
    sequences = MOT16Sequences(seq_name, args.data_dir, load_seg=False)

    tracker_number=args.tracker

    if tracker_number==1:
        tracker=IoUAssignmentTracker(obj_detect,inactive_pateince=inactive_pateince)

    elif tracker_number==2:
        tracker=HungarianIoUTracker(obj_detect,inactive_pateince=inactive_pateince)

    elif tracker_number==3:
        # #Reid Network
        feature_extractor = torch.load(args.reid_network_path)
        tracker=ReIDTracker(feature_extractor,obj_detect,inactive_pateince=inactive_pateince)
    else:
        raise NotImplementedError("Unknown tracker number: "+str(tracker_number)+", available trackers are (1,2,3)\n run with --help option for more information")

    run_tracker(tracker=tracker,sequences=sequences,output_dir=output_dir)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)