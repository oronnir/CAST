import cv2
import json
import numpy as np
from Animator.consolidation_api import CharacterDetectionOutput, CharacterConsolidationOutput
from Detector.background_cropper import Range
from datetime import timedelta
from dateutil.parser import parse
from Detector.detector_wrapper import DetectorWrapper


class ShotsLoader:
    def __init__(self, video_path, detection_json_path, grouping_json_path, insights_json_path):
        self.video_path = video_path
        self.detection_json_path = detection_json_path
        self.grouping_json_path = grouping_json_path
        self.detections = CharacterDetectionOutput.read_from_json(self.detection_json_path)
        self.keyframe_to_detections = self.detections.consolidate_by_keyframes()
        with open(insights_json_path, 'r') as vi_insights:
            self.insights_full = json.load(vi_insights)
        self.insights = self.insights_full['videos'][0]['insights']
        self.shots = self.insights['shots']
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.cv2.CAP_PROP_FPS)
        self.num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration_in_seconds = self.num_frames/self.fps
        cap.release()
        cv2.destroyAllWindows()

    def load_shots(self, instance_focus, should_play):
        shot_range = self.instance_to_range(instance_focus['instances'][0])
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        second = shot_range.start
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)  # optional
        success, image = cap.read()
        while success and second <= shot_range.end:
            second += 1/self.fps
            cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            success, frame = cap.read()
            frames.append(frame)
            if should_play:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()
        return frames

    def get_query_appearances(self, shot_query):
        """creates a shot with all appearances and their conjunct instances"""
        query_detections = []
        for keyframe in shot_query['keyFrames']:
            instance = keyframe['instances'][0]
            kf_detected_instances = self.keyframe_to_detections.get(instance['thumbnailId'], None)
            query_detection = dict(kf_detections=kf_detected_instances,
                                   keyframe_id=instance['thumbnailId'],
                                   kf_instance=instance)
            query_detections.append(query_detection)
        return query_detections

    def klt(self, frames):
        """
        run the KLT tracker on shot frames
        :return: tracks
        """
        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)

        # Parameters for Lucas Kanade optical flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))
        old_frame = frames[0]
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        frame_no = -1
        for frame in frames:
            frame_no += 1
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if int(frame_no) % 5 == 0:
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

            img = cv2.add(frame, mask)
            cv2.imshow('frame', img)
            k = cv2.waitKey(int(self.fps)) & 0xff
            if k == 27:
                break

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

    def meanshift(self, frames, detections):
        frame = frames[0]

        # setup initial location of window
        rect1 = detections[0]['kf_detections'][0].Rect
        c = rect1.X
        r = rect1.Y
        w = rect1.Width
        h = rect1.Height
        track_window = (c, r, w, h)

        # set up the ROI for tracking
        roi = frame[r:r + h, c:c + w]
        hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        for frame_index in range(len(frames)):
            frame = frames[frame_index]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # apply meanshift to get the new location
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)

            # Draw it on image
            x, y, w, h = track_window
            img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
            cv2.imshow('img2', img2)

            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
            else:
                cv2.imwrite(chr(k) + ".jpg", img2)

        cv2.destroyAllWindows()

    @staticmethod
    def to_total_seconds(date_time):
        return timedelta(hours=date_time.hour,
                         minutes=date_time.minute,
                         seconds=date_time.second,
                         microseconds=date_time.microsecond)\
            .total_seconds()

    @staticmethod
    def instance_to_range(instance):
        instance_start = ShotsLoader.to_total_seconds(parse(instance['start'], fuzzy=True))
        instance_end = ShotsLoader.to_total_seconds(parse(instance['end'], fuzzy=True))
        return Range(instance_start, instance_end)

    @staticmethod
    def range_to_instance(time_range):
        start = timedelta(seconds=time_range.start)
        end = timedelta(seconds=time_range.end)
        return {'adjustedStart': start.__str__(),
                'adjustedEnd': end.__str__(),
                'start': start.__str__(),
                'end': end.__str__()}


if __name__ == '__main__':
    detector = DetectorWrapper()
    keyframes_in = "???\\TrackerData\\Huffless\\out\\Keyframes"
    detections_out = "???\\TrackerData\\Huffless\\out\\detections"
    detector.detect_and_featurize(keyframes_in, detections_out)

    vid_path = r"???\TrackerData\Daffy Duck and the Dinosaur (1939)-cfVna0EA_L8\Daffy Duck and the Dinosaur (1939)-cfVna0EA_L8.mp4"
    insights_path = r"???\TrackerData\Daffy Duck and the Dinosaur (1939)-cfVna0EA_L8\insights\insights.json"
    detections_path = r"???\TrackerData\Daffy Duck and the Dinosaur (1939)-cfVna0EA_L8\detections\animationdetectionoutput.json"
    grouping_path = None
    shot_loader = ShotsLoader(vid_path, detections_path, grouping_path, insights_path)
    shot_frames = None
    for j in [22, 66, 92]:
        shot_focus_id = j
        shot_focus = shot_loader.shots[shot_focus_id]
        shot_detections = shot_loader.get_query_appearances(shot_focus)
        if len(shot_detections) == 0:
            continue
        shot_frames = shot_loader.load_shots(shot_focus, should_play=True)
        shot_loader.meanshift(shot_frames, shot_detections)
