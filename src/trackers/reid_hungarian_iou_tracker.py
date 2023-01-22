import torch
import collections
import numpy as np
import motmetrics as mm

from trackers.basic_tracker import BasicTracker,BasicTrack
from scipy.optimize import linear_sum_assignment as linear_assignment
from utils.utils import compute_distance_matrix,cosine_distance,ltrb_to_ltwh,get_crop_from_boxes


mm.lap.default_solver = 'lap'
_UNMATCHED_COST = 255.0


class ReIDTracker(BasicTracker):
    def __init__(self, feature_extractor, *args, **kw):
        super().__init__(*args, **kw)
        self.feature_extractor = feature_extractor

    def add(self, new_boxes, new_scores, new_features):
        """Initializes new Track objects and saves them."""
        num_new = len(new_boxes)
        for i in range(num_new):
            self.tracks.append(ReidTrack(
                new_features[i],
                new_boxes[i],
                new_scores[i],
                self.track_num + i,
                self.im_index,
            ))
        self.track_num += num_new

    def data_association(self, boxes, scores, frame):
        crops = get_crop_from_boxes(boxes, frame)
        pred_features = self.compute_reid_features(self.feature_extractor, crops).cpu().clone()
        
        if self.tracks or self.previous_tracks:
            new_track_list=self.tracks+self.previous_tracks

            
            track_ids = [t.id for t in new_track_list]
            track_boxes = torch.stack([t.box for t in new_track_list], axis=0)
            track_features = torch.stack([t.get_feature() for t in new_track_list], axis=0)
            
            distance = self.compute_distance_matrix(track_features, pred_features,
                                                    track_boxes, boxes, metric_fn=cosine_distance)

            # Perform Hungarian matching.
            row_idx, col_idx = linear_assignment(distance)           
            
            remove_track_ids = [t.id for t in new_track_list]
            new_track_idx=[i for i in range(len(boxes))]
            for track_idx,box_idx in zip(row_idx,col_idx):
                if distance[track_idx,box_idx] < _UNMATCHED_COST:
                    new_track_list[track_idx].box=boxes[box_idx]
                    new_track_list[track_idx].add_feature(pred_features[box_idx])
                    new_track_idx.remove(box_idx)
                    remove_track_ids.remove(track_ids[track_idx])
                    
            self.tracks = [t for t in new_track_list
                           if t.id not in remove_track_ids]

            new_boxes =  [boxes[i] for i in new_track_idx]  # <-- needs to be filled.
            new_scores = [scores[i] for i in new_track_idx] # <-- needs to be filled.
            new_features = [pred_features[i] for i in new_track_idx] # <-- needs to be filled.
            
            self.previous_tracks=[t for t in new_track_list
                           if t.id  in remove_track_ids]

            self.add(new_boxes, new_scores, new_features)
        else:
            # No tracks exist.
            self.add(boxes, scores, pred_features)

    def step(self, frame):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        # object detection
        boxes, scores = self.obj_detect.detect(frame['img'])

        self.data_association(boxes, scores, frame['img'])

        # results
        for t in self.tracks:
            t.frame_number=self.im_index
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate([t.box.cpu().numpy(), np.array([t.score])])

        self.im_index += 1  
        self.previous_tracks=[pt for pt in self.previous_tracks if self.im_index-pt.frame_number<=self.inactive_pateince]

    def compute_reid_features(self, model, crops):
        f_ = []
        model.eval()
        with torch.no_grad():
            for data in crops:
                img = data.cuda()
                features = model(img)
                features = features.cpu().clone()
                f_.append(features)
            f_ = torch.cat(f_, 0)
            return f_
 
    def compute_distance_matrix(self, track_features, pred_features, track_boxes, boxes, metric_fn, alpha=0.0):
        UNMATCHED_COST = 255.0
        # Build cost matrix.
        iou_track_boxes = ltrb_to_ltwh(track_boxes)
        iou_boxes = ltrb_to_ltwh(boxes)
        distance = mm.distances.iou_matrix(iou_track_boxes, iou_boxes.numpy(), max_iou=0.5)
        # distance = mm.distances.iou_matrix(track_boxes.numpy(), boxes.numpy(), max_iou=0.5)

        appearance_distance = compute_distance_matrix(track_features, pred_features, metric_fn=metric_fn)
        appearance_distance = appearance_distance.numpy() * 0.5
        # return appearance_distance

        assert np.alltrue(appearance_distance >= -0.1)
        assert np.alltrue(appearance_distance <= 1.1)

        combined_costs = alpha * distance + (1-alpha) * appearance_distance

        # Set all unmatched costs to _UNMATCHED_COST.
        distance = np.where(np.isnan(distance), UNMATCHED_COST, combined_costs)
        return distance

class ReidTrack(BasicTrack):
    def __init__(self, feature, *args, **kw):
        super().__init__(*args, **kw)
        self.feature = collections.deque([feature])
        self.max_features_num = 10
    

    def add_feature(self, feature):
        """Adds new appearance features to the object."""
        self.feature.append(feature)
        if len(self.feature) > self.max_features_num:
            self.feature.popleft()

    def get_feature(self):
        if len(self.feature) > 1:
            feature = torch.stack(list(self.feature), dim=0)
        else:
            feature = self.feature[0].unsqueeze(0)
        return feature.mean(0, keepdim=False)




    
