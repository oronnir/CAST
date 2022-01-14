class Detection:
    def __init__(self, frame_id, detection_id, x1, y1, width, height, some_binary=0, f1=-1, f2=-1, f3=-1):
        self.frame_id = frame_id
        self.detection_id = detection_id
        self.x1 = x1
        self.w = width
        self.y1 = y1
        self.h = height
        self.some_binary = some_binary
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3

    @staticmethod
    def parse(line):
        frame_id, detection_id, x1, y1, w, h, some_binary, f1, f2, f3 = line.split(',')
        return Detection(int(frame_id), int(detection_id), int(x1), int(y1), int(w), int(h), 0, -1, -1, -1)

    def __repr__(self):
        return f"{self.frame_id},{self.detection_id},{self.x1},{self.y1},{self.w},{self.h},{self.some_binary}," \
               f"{self.f1},{self.f2},{self.f2}"
