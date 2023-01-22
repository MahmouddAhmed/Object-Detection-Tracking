import numpy as np
import motmetrics as mm
from utils.utils import ltrb_to_ltwh
from trackers.basic_tracker import BasicTracker
from scipy.optimize import linear_sum_assignment as linear_assignment




_UNMATCHED_COST = 255.0
class HungarianIoUTracker(BasicTracker):
    """
    This tracker utilizes Bipartite matching such that each track in a previous frame is matched with eactly
    one track in the current frame if the Iou is above o.5
    """

    def data_association(self, boxes, scores):
        if self.tracks or self.previous_tracks:

            new_track_list=self.tracks+self.previous_tracks
            track_boxes = np.stack([t.box.numpy() for t in new_track_list], axis=0)
            
            
            # Build cost matrix.
            iou_track_boxes = ltrb_to_ltwh(track_boxes)
            iou_boxes = ltrb_to_ltwh(boxes)
            distance = mm.distances.iou_matrix(
                iou_track_boxes, iou_boxes.numpy(), max_iou=0.5)

            # Set all unmatched costs to _UNMATCHED_COST.
            distance = np.where(np.isnan(distance), _UNMATCHED_COST, distance)            

            # Perform Hungarian matching.
            row_idx, col_idx = linear_assignment(distance)
    
            # Update existing tracks and remove unmatched tracks.
            # 1. costs == _UNMATCHED_COST -> remove.
            # 2. tracks that have no match -> remove.
            remove_track_ids = [t.id for t in new_track_list]
            new_track_idx=[i for i in range(len(boxes))]
            for track_idx,box_idx in zip(row_idx,col_idx):
                if distance[track_idx,box_idx] < _UNMATCHED_COST:
                    new_track_list[track_idx].box=boxes[box_idx]
                    new_track_idx.remove(box_idx)
                    remove_track_ids.remove(new_track_list[track_idx].id)
                    
            self.tracks = [t for t in new_track_list
                           if t.id not in remove_track_ids]

            # Add new tracks.
            new_boxes = [boxes[i] for i in new_track_idx] 
            new_scores = [scores[i] for i in new_track_idx]


            # Add the unmatched tracks to the previous tracks list
            # Remove tracks that have been matched from the previous tracks list
            self.previous_tracks=[t for t in new_track_list
                           if t.id  in remove_track_ids]
  
            self.add(new_boxes, new_scores)
        else:
            # No tracks exist.
            self.add(boxes, scores)