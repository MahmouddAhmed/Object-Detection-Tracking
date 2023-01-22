import torch
import numpy as np
class BasicTracker:
    """
    A Basic Object Tracker Without any data association between frames,
    Such that every Object Detected is considered a different track.
    ...

    Attributes
    ----------
    obj_detect : Object
        An Object Detector that has a detect method that takes as input images and returns the detected
        box cordinates and class scores for each detected object
    tracks : List
        Stores all detected tracks
    track_num : Int
        The total number of detected tracks
    im_index : int
        The number of frames (images) processed
    results : Dictionary<int,int,List>
        Accumulates the result of each track through all frames and the associated bounding boxes and scores.
        In the form of [Track ID][Frame Number][Box,Score]
    previous_tracks : List
        keeps tracks of previous tracks that have not been assigned for a number of frames
    inactive_patience: int
        The number of frames that a track will stay in the previous tracks without being matched after that the 
        tracks will not be taken into consideration
    """    

    def __init__(self, obj_detect,inactive_pateince=60):
        self.obj_detect = obj_detect
        self.tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}
        self.previous_tracks=[]
        self.inactive_pateince=inactive_pateince

    def reset(self, hard=True):
        self.tracks = []
        self.previous_tracks=[]
        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    def add(self, new_boxes, new_scores):
        """Initializes new Track objects and saves them."""
        num_new = len(new_boxes)
        for i in range(num_new):
            self.tracks.append(BasicTrack(
                new_boxes[i],
                new_scores[i],
                self.track_num + i,
                self.im_index 

            ))
        self.track_num += num_new

    def get_pos(self):
        """Get the positions of all active tracks."""
        if len(self.tracks) == 1:
            box = self.tracks[0].box
        elif len(self.tracks) > 1:
            box = torch.stack([t.box for t in self.tracks], 0)
        else:
            box = torch.zeros(0).cuda()
        return box

    def data_association(self, boxes, scores):
        self.previous_tracks=self.previous_tracks+self.tracks
        self.tracks = []
        self.add(boxes, scores)

    def step(self, frame):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        # object detection
        boxes, scores = self.obj_detect.detect(frame['img'])

        self.data_association(boxes, scores)

        # results
        for t in self.tracks:
            t.frame_number=self.im_index
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate([t.box.cpu().numpy(), np.array([t.score])])

        self.im_index += 1
        self.previous_tracks=[pt for pt in self.previous_tracks if self.im_index-pt.frame_number<=self.inactive_pateince]

    def get_results(self):
        return self.results




class BasicTrack(object):
    """A Basic class that contains all necessary for every individual track."""

    def __init__(self, box, score, track_id,frame_number=-1):
        self.id = track_id
        self.box = box
        self.score = score
        self.frame_number=frame_number
