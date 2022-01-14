import os
from Animator.bbox_grouper_api import CharacterDetectionOutput
from Tracker.sort_wrapper import Detection


class SortFormatter:
    @staticmethod
    def create_ini(working_dir, vid_name, folder_name, frame_rate, num_frames, width, height):
        ini_file_path = os.path.join(working_dir, 'seqinfo.ini')
        if os.path.isfile(ini_file_path):
            return
        ini_content = f"[Sequence]\nname={vid_name}\nimDir={folder_name}\nframeRate={frame_rate}\n" \
                      f"seqLength={num_frames}\nimWidth={width}\nimHeight={height}\nimExt=.jpg\n\n"
        with open(ini_file_path, 'w') as ini_file:
            ini_file.write(ini_content)

    @staticmethod
    def vi_to_sort(vi_detections: CharacterDetectionOutput, sort_det_file_path: str):
        lines = []
        box_in_frame = 0
        frame_index = 0
        with open(sort_det_file_path, 'w') as sort_format:
            for vi_det in vi_detections.CharacterBoundingBoxes:
                if frame_index != vi_det.KeyFrameIndex:
                    frame_index += 1
                    box_in_frame = 0
                sort_det = Detection(frame_id=vi_det.KeyFrameIndex, detection_id=box_in_frame,
                                     x1=vi_det.Rect.X, y1=vi_det.Rect.Y, width=vi_det.Rect.Width,
                                     height=vi_det.Rect.Height)
                lines.append(sort_det.__repr__()+'\n')
                box_in_frame += 1
            sort_format.writelines(lines)
        return
