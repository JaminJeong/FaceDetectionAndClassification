import cv2
import face_recognition

import sys
sys.path.append("../")

from common.camera import VideoCamera
from common.FaceDetector import FaceDetector

class FaceDetectorCV(FaceDetector):
    def __init__(self):
        super()._init()

    def DetectROI(self, frame):
        h, w, c = frame.shape
        rgb_small_frame = frame[:, :, ::-1]
        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        # self.face_locations = face_recognition.face_locations(rgb_small_frame, model='cnn') #very slow
        print(f"face_locations : {self.face_locations}")

        for value in self.face_locations:
            self.faceDetectedResult.boxes.append([
                float(value[2] / h), float(value[3] / w),
                float(value[0] / h), float(value[1] / w)])
            self.faceDetectedResult.scores.append(1.0)

        return self.faceDetectedResult

def draw_opencv(frame, face_locations, face_names, is_draw_text=False, is_draw_text_fill=False):
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        if is_draw_text:
            if is_draw_text_fill:
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            else:
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255))
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    return frame

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog="inference",
                                     description="detect face", add_help=True)
    parser.add_argument('-i', '--INPUTVIDEO', help='input video.', required=True)
    args = parser.parse_args()

    cam = VideoCamera(args.INPUTVIDEO) #'./tensorflow-face-detection/media/test.mp4'
    # cam = VideoCamera('./tensorflow-face-detection/media/test.mp4')

    detector = FaceDetectorCV()
    while True:
        frame = cam.get_frame()
        dt_result = detector.DetectROI(frame)

        draw_image = draw_opencv(frame.copy(),
                                 detector.face_locations,
                                 ["face" for x in range(len(detector.face_locations))]                                 )
        # show the frame
        cv2.imshow("Frame", draw_image)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    print('finish')
