import cv2
import numpy as np
import itertools

from numpy.random import randint
from scipy.spatial import distance_matrix
from conv_net import *
from settings import *
from utils import *

# redetect flag
redetect = False

# global pool of detectors and descriptors
detector_pool = [cv2.FeatureDetector_create(DETECTOR) for i in range(MAX_FACES)]
descriptor_pool = [cv2.DescriptorExtractor_create(DESCRIPTOR) for i in range(MAX_FACES)]

class FaceTracker(object):
    """
    Lucas Kalande feature detector
    """

    def __init__(self, frame, patch_location, gender, lk_params, feature_params):
        
        global detector_pool, descriptor_pool
        
        # assign the object properties
        self.face_id = randint(1000)
        self.gender = gender
        print 'Face_id {} initialized for tracking'.format(self.face_id)
        self.detector_method = DETECTOR
        self.fg_mask = np.zeros_like(frame)
        x,y,w,h = patch_location
        self.fg_mask[y:y+h, x:x+w] = 255
        self.bg_mask = 255 - self.fg_mask
        self.lk_params = lk_params
        self.feature_params = feature_params

        # check if we have detected more faces than `MAX_FACES` 
        # If not, then assign a detector from the pool, else
        # create a new one
        if detector_pool != []:
            self.detector = detector_pool.pop()
        else:
            self.detector = cv2.FeatureDetector_create(self.detector_method)
        
        if descriptor_pool is not []:
            self.descriptor = descriptor_pool.pop()
        else:
            self.descriptor = cv2.DescriptorExtractor_create(DESCRIPTOR)
        
        self.image_dims = frame.shape
        self.matcher = cv2.DescriptorMatcher_create(MATCHER)
        
        # detect  goreground (face) keypoints
        self.fg_keypoints = self.detector.detect(frame, mask=self.fg_mask)
        
        # convert the keypoints from opencv keypoint class to cartesian points
        self.old_points = np.array([kp.pt for kp in self.fg_keypoints], 
                                   dtype=np.float32).reshape(-1,1,2)
        self.new_points = self.old_points
        
        # default bounding box
        self.bounding_box = (0,0,0,0)
        
    def update_keypoints(self, frame, patch_location):
        """
        Updates the object with the keypoints detected after
        re detection
        """

        x,y,w,h = patch_location
        
        # detect only the foreground keypoints
        fg_mask = np.zeros_like(frame)
        fg_mask[y:y+h, x:x+w] = 1
        new_features = self.detector.detect(frame, mask=fg_mask)
        
        new_features = np.array([kp.pt for kp in new_features], dtype=np.float32)
        
        existing_features = self.old_points.reshape(-1,2)
        
        # from the existing points, filterout those which are not in the
        # updated face area. This will reduce tracking any backround
        # keypoint
        good_indices = in_rect(existing_features, (x,y), (x+w,y+h))
        good_features = np.array([f for f in itertools.compress(existing_features,
                                                                good_indices)])
        
        try:
            good_features = np.vstack((new_features, good_features))
            self.old_points = good_features.reshape(-1,1,2)
        except Exception as e:
            pass
    
    def track(self, old_frame, new_frame):
        """
        This method gives an updated bounding box after each frame based
        on Lucas-Kanade optical flow tracking
        """
        
        global redetect
        self.old_points = np.reshape(self.old_points, (-1,1,2))
        
        # forward detection
        self.new_points, st, err = cv2.calcOpticalFlowPyrLK(old_frame, 
                                                            new_frame, 
                                                            self.old_points, 
                                                            None, 
                                                            **self.lk_params)
        # backward redetection
        old_points_recon, st, err = cv2.calcOpticalFlowPyrLK(new_frame, 
                                                             old_frame, 
                                                             self.new_points, 
                                                             None, 
                                                             **self.lk_params)
        
        # discard the points which have even a single pixel displacement
        # after the forward-backward error detection
        d = abs(self.old_points - old_points_recon).reshape(-1,2).max(-1)
        good_points = d < 1
        self.new_points = np.array([pt for pt in itertools.compress(self.new_points,
                                                                    good_points)])
        
        # at least two keypoints are neede for tracking
        if len(self.new_points.shape) < 2:
            redetect = True
            return (0,0,0,0)

        #self.remove_outliers()

        # update the new points
        self.old_points = self.new_points

        # get the updated bounding box
        x,y,w,h = cv2.boundingRect(self.new_points)
        self.bounding_box = (x,y,w,h)
        
        return (x,y,w,h)
    

    def draw_features(self, image):
        """
        Draws a point in the image for each kepoint
        """
        
        for x,y in self.new_points.reshape(-1,2):
            cv2.circle(image, (x,y), 2, (255,0,255), 2)
        return image
    

    def is_weak(self):
        """
        This method checks for the consistency of the tracker
        If the tracker goes out of the frame or it tracks less than
        a minimum number of keypoints, then the tracker is
        deemed unusable
        """

        # set the minimum number of keypooints
        keypoint_threshold = 20 if self.detector_method == 'FAST'\
                else 5
        
        # check if the tracker has less than minimum keypoints to track
        c1 = self.old_points.shape[0] < keypoint_threshold
        
        x,y,w,h = self.bounding_box
        row, col = self.fg_mask.shape
        
        # check if the window is out of the frame
        c2 = x >= col-1 or x < 0
        c3 = y >= row-1 or y < 0
        c4 = x+w >= col-1
        c5 = y+h >= row-1
        
        return c1+c2+c3+c4+c5


    def set_color(self, color):
        """
        Set a specific color to the bounding box
        """
        self.color = color

    def get_color(self):
        """
        Get the color of the bounding box
        """
        return self.color

    def remove_outliers(self):
        global redetect
        points = self.new_points.reshape(-1,2)
        dist_matrix = distance_matrix(points, points, p=2)
        points = map(list, [p for p in points])
        sum_of_dist = sum(dist_matrix)
        good_points = [ abs(sum_of_dist - np.mean(sum_of_dist)) < 2*np.std(sum_of_dist)]
        
        for p,g in zip(points, good_points[0]):
            if not g:
                points.remove(p)
        if len(points) < 20:
            redetect = True
        self.new_points = np.array(points).reshape(-1, 1, 2)


if __name__ == '__main__':
    
    # initialize the convolution neural network
    conv = ConvNet(CONV_CONFIG, NNET_CONFIG, INPUT_SIZE)
    conv.load_params(CONV_PARAMS)

    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture("/home/vinayak/Videos/bw.mp4")
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
     
    val, image = cap.read()
    #image = cv2.pyrDown(image)
    image = cv2.resize(image, (320,240))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    prev_gray = gray
    rows, cols = gray.shape
    
    # some settings variables
    re_detect_count = 5 # run detection after every $ steps
    frames_without_detection = 200
    frame_idx = 1
    track_len = 10

    face_cascade_frontal = cv2.CascadeClassifier(FRONTAL_FACE)
    #face_cascade_profile = cv2.CascadeClassifier(PROFILE_FACE)
   
    tracker_objects = []
    
    while True:

        val, image = cap.read()
        if image is None:
            break
        #image = cv2.pyrDown(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # check if we need to detect
        if frame_idx % re_detect_count == 0 or redetect:
            print frame_idx
            redetect = False

            frontal_faces = face_cascade_frontal.detectMultiScale(gray, 1.3, 5)
            #profile_faces_left = face_cascade_profile.detectMultiScale(np.fliplr(gray), 1.3,5)
            #profile_faces_left = [(cols-(x+w),y,w,h) for (x,y,w,h) in profile_faces_left]
            faces = list(frontal_faces) #+ list(profile_faces_left) + list(profile_faces_right)
            
            frame_idx += 1

            for f in faces:
                
                x,y,w,h = f
                face_image = gray[y:y+h, x:x+w]
                face_image = cv2.resize(face_image, INPUT_SIZE)/255.
                
                # predict the gender
                pred = conv.predict(np.array([face_image]))
                pred = np.argmax(pred)
                gender = 'Male' if pred ==1 else 'Female'
                
                # check if the face detected is an existing one or not
                # If it is an existing one, update the corresponding tracker
                # object
                track_obj, status = is_existing_face(gray, tracker_objects, f)
                if status:
                    track_obj.update_keypoints(gray, f)
                    track_obj.gender = gender
                    draw_rectangle(image, track_obj.bounding_box, color=track_obj.get_color())
                else: 
                    # if the detectd face is a new one, create a tracker object
                    track_obj = FaceTracker(gray, f, gender, lk_params, feature_params)
                    track_obj.set_color((randint(0,255), randint(0,255), randint(0,255)))
                    tracker_objects.append(track_obj)
                
                cv2.putText(image, track_obj.gender, (max(x-10,0), max(y-10, 0)), 
                            cv2.FONT_HERSHEY_PLAIN, 2, 255)

            if len(faces) == 0:
                for track_obj in tracker_objects:
                    draw_rectangle(image, track_obj.bounding_box, color=track_obj.get_color())
                    cv2.putText(image, tracker.gender, (max(x-10,0), max(y-10, 0)),
                                cv2.FONT_HERSHEY_PLAIN, 2, 255)

        
        # no detection, only tracking
        else:
            frame_idx += 1
            print frame_idx

            for tracker in tracker_objects:
                # check if the tracker is a weak one, if yes, delete it and
                # return the detector and descriptor to the global pool
                if tracker.is_weak() or is_redundant(tracker, tracker_objects):
                    tracker_objects.remove(tracker)
                    detector_pool.append(tracker.detector)
                    descriptor_pool.append(tracker.descriptor)
                    continue
                
                # if the tracker is not a weak one, get the current
                # bounding box
                (x,y,w,h) = tracker.track(prev_gray, gray)
                image = draw_rectangle(image, (x,y,w,h), color=tracker.get_color())
                cv2.putText(image, tracker.gender, (max(x-10,0), max(y-10, 0)), 
                            cv2.FONT_HERSHEY_PLAIN, 2, 255)

        prev_gray = gray
        cv2.imshow("image", image)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

