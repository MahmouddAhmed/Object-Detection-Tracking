import numpy as np
import motmetrics as mm

from utils.utils import ltrb_to_ltwh
from trackers.basic_tracker import BasicTracker


class IoUAssignmentTracker(BasicTracker):
    '''
    A Basic tracker that matches tracks between two consecutive frames using IOU,
    However it does not provide a unique assignment.    
    '''

    def data_association(self, boxes, scores):
        '''
        Builds a cost matrix as follows: Given  n  tracks and  m  detected boxes, 
        the result will be a  n×m  matrix with each being  1−IoU(track,box)  for  IoU>=0.5  
        and np.nan otherwise. Then, each track is extended with the best box in terms of costs. 
        All predicted boxes, that don't overlap, i.e.,  IoU<0.5 , at all with the tracks start new tracks.
        '''
        
        if self.tracks or self.previous_tracks:
            track_boxes = np.stack([t.box.numpy() for t in self.tracks], axis=0)
            
            iou_track_boxes = ltrb_to_ltwh(track_boxes)
            iou_boxes = ltrb_to_ltwh(boxes)
            distance = mm.distances.iou_matrix(
                iou_track_boxes, iou_boxes.numpy(), max_iou=0.5)

            # update existing tracks
            remove_track_ids = []
            for t, dist in zip(self.tracks, distance):
                if np.isnan(dist).all():
                    remove_track_ids.append(t.id)
                else:
                    match_id = np.nanargmin(dist)
                    t.box = boxes[match_id]
            self.tracks = [t for t in self.tracks
                           if t.id not in remove_track_ids]

            # add new tracks
            new_boxes = []
            new_scores = []
            for i, dist in enumerate(np.transpose(distance)):
                if np.isnan(dist).all():
                    new_boxes.append(boxes[i])
                    new_scores.append(scores[i])
            self.add(new_boxes, new_scores)

        else:
            self.add(boxes, scores)