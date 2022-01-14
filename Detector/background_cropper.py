class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __repr__(self):
        return 'Range({}, {})'.format(self.start, self.end)

    def is_valid(self):
        return self.end > self.start

    def is_conjunct(self, other, left_inclusive=True, right_inclusive=True):
        if other is None:
            return False

        return (self.start < other.end or (left_inclusive and self.start == other.end)) and \
               (self.end > other.start or (right_inclusive and self.end == other.start))

    def except_conjunct(self, other):
        """ return the segments in self that are not in other """
        # case they are foreign
        if not self.is_conjunct(other):
            return [self]

        # other contain self
        if self.start >= other.start and self.end <= other.end:
            return []

        # partial conjunction - self starts first
        if self.start < other.start and self.end < other.end:
            return [Range(self.start, other.start)]

        # partial conjunction - other starts first
        if self.start > other.start and self.end > other.end:
            return [Range(other.end, self.end)]

        # other is contained in self so we got two segments
        return [Range(self.start, other.start), Range(other.end, self.end)]

    def conjunct(self, other):
        if not self.is_conjunct(other):
            return Range(0, 0)
        else:
            start = max(self.start, other.start)
            end = min(self.end, other.end)
            return Range(start, end)


class SubFrame(object):
    def __init__(self, top, left, width, height):
        self.top = top
        self.left = left
        self.width = width
        self.height = height

    def __repr__(self):
        return 'SubFrame({}, {}, {}, {})'.format(self.top, self.left, self.width, self.height)

    def __eq__(self, other):
        return self.top == other.top \
               and self.left == other.left \
               and self.width == other.width \
               and self.height == other.height

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return 1*self.top + 10**4*self.left + 10**8*self.width + 10**12*self.height

    @property
    def center(self):
        x_center = self.left + self.width/2
        y_center = self.top + self.height/2
        return x_center, y_center

    def euclidean_distance(self, other_x, other_y):
        self_x, self_y = self.center
        return ((self_x - other_x) ** 2 + (self_y - other_y) ** 2) ** 0.5

    def get_centered_subframe(self, subframes):
        distances_from_subframe_center = [self.euclidean_distance(*box.center) for box in subframes]
        idx_max, val_max = min(enumerate(distances_from_subframe_center))
        return idx_max, val_max

    def overshadowing(self, other):
        return self.top <= other.top and \
               self.left <= other.left and \
               self.top + self.height >= other.top + other.height and \
               self.left + self.width >= other.left+other.width

    @staticmethod
    def _box_splitter(frame, boxes, min_subframe_width, min_subframe_height, min_subframe_area):
        # in case the potential subframe is too small
        if frame.height < min_subframe_height \
                or frame.width < min_subframe_width \
                or frame.width*frame.height < min_subframe_area:
            return set()

        # copy boxes to a new list with new instances
        subframes = [SubFrame(sf.top, sf.left, sf.width, sf.height) for sf in boxes]

        # stopping criteria
        if len(subframes) == 0:
            return {frame}

        largest_background_boxes = set()

        # pick the most centered box relative to the subframe to improve the average case
        centered_idx, _ = frame.get_centered_subframe(subframes)
        current_box = subframes.pop(centered_idx)

        # split horizontally
        horizontal_frames = Range(frame.left, frame.left + frame.width).except_conjunct(Range(current_box.left, current_box.left + current_box.width))
        for horizontal_range in horizontal_frames:
            subframe_width = horizontal_range.end - horizontal_range.start
            horizontal_frame = SubFrame(frame.top, horizontal_range.start, subframe_width, frame.height)
            largest_background_boxes |= SubFrame._box_splitter(horizontal_frame, subframes, min_subframe_width, min_subframe_height, min_subframe_area)

        # split vertically
        vertical_frames = Range(frame.top, frame.top + frame.height).except_conjunct(Range(current_box.top, current_box.top + current_box.height))
        for vertical_range in vertical_frames:
            subframe_height = vertical_range.end - vertical_range.start
            vertical_frame = SubFrame(vertical_range.start, frame.left, frame.width, subframe_height)
            largest_background_boxes |= SubFrame._box_splitter(vertical_frame, subframes, min_subframe_width, min_subframe_height, min_subframe_area)

        return largest_background_boxes

    @staticmethod
    def consolidate_overlapping_bboxes(bboxes):
        """ eliminate bounding box eclipsing to reduce effective complexity """
        redundant_boxes = set()
        for bbox_i in bboxes:
            for bbox_j in bboxes:
                if bbox_i == bbox_j:
                    continue
                if bbox_i.overshadowing(bbox_j):
                    redundant_boxes.add(bbox_j)
        nonredundant_boxes = bboxes.copy()
        for redundant_box in redundant_boxes:
            nonredundant_boxes.remove(redundant_box)
        return nonredundant_boxes

    @staticmethod
    def get_background_subframes(frame, bboxes, min_subframe_width=1, min_subframe_height=1, min_subframe_area=1,
                                 max_unique_boxes=15):
        """
        Return all background bounding boxes (as a Subframe set) that do not intersect with the bboxes in the frame.
        The complexity of the underlying problem is o(4^n). Therefore, in some cases the function will avoid solving
        this NP problem.
        :param max_unique_boxes: cap on the number of boxes for runtime optimization.
        :param frame: the original image dimensions (also a SubFrame)
        :param bboxes: the bounding boxes stored in either a list or a set of SubFrames
        :param min_subframe_width: the minimal valid width in pixels for a background box
        :param min_subframe_height: the minimal valid height in pixels for a background box
        :param min_subframe_area: the minimal valid area in pixels^2 for a background
        :return: list of background subframes
        """
        # try to merge redundant bboxes
        if type(bboxes) != set:
            bboxes = set(bboxes)

        # verify unique input size.
        unique_bboxes = SubFrame.consolidate_overlapping_bboxes(bboxes)

        # the unbounded problem's complexity is o(4^n).
        input_num_unique_boxes = len(unique_bboxes)
        if input_num_unique_boxes > max_unique_boxes:
            print('the number of unique bboxes ({}) exceeded the feasible maximum ({})'
                  .format(input_num_unique_boxes, max_unique_boxes))
            return set()
        return SubFrame._box_splitter(frame, unique_bboxes, min_subframe_width, min_subframe_height, min_subframe_area)
