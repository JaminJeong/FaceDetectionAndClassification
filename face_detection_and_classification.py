import cv2
import numpy as np
from common.camera import VideoCamera
from common.Face import Face

if __name__ == '__main__':
    def draw_cv(frame, face_locations, face_names):
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255))
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return frame

    import argparse

    parser = argparse.ArgumentParser(prog="inference",
                                     description="detect face", add_help=True)
    parser.add_argument('-i', '--INPUTVIDEO', help='input video.', required=True)
    parser.add_argument('-f', '--FACEDIR', help='folder of face images.', required=True)
    args = parser.parse_args()

    cam = VideoCamera(args.INPUTVIDEO) #'./tensorflow-face-detection/media/test.mp4'
    # cam = VideoCamera('./tensorflow-face-detection/media/test.mp4')

    from face_detection.face_detection_opencv import FaceDetectorCV
    detector_cv = FaceDetectorCV()
    from face_recognitions.face_recog_opencv import FaceRecogCV
    recog_cv = FaceRecogCV(args.FACEDIR)

    while True:
        frame = cam.get_frame()
        # frame = np.array(frame)
        # print(f"frame shape : {frame.shape}" )
        dt_result = detector_cv.DetectROI(frame)
        recog_cv.Recog(frame, detector_cv.face_locations, dt_result)
        draw_image = draw_cv(frame.copy(), detector_cv.face_locations, dt_result.names)

        # show the frame
        # cv2.imshow("Frame", frame)
        cv2.imshow("Frame", draw_image)

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    print('finish')
