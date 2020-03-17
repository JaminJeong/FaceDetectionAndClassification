import cv2
import numpy as np
from common.camera import VideoCamera
from common.Face import Face

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog="inference",
                                     description="detect face", add_help=True)
    parser.add_argument('-i', '--INPUTVIDEO', help='input video.', required=True)
    parser.add_argument('-f', '--FACEDIR', help='folder of face images.', required=True)
    parser.add_argument('-o', '--OUTPUTDIR', help='output video.', required=True)
    args = parser.parse_args()

    out = None
    cam = VideoCamera(args.INPUTVIDEO) #'./tensorflow-face-detection/media/test.mp4'
    # cam = VideoCamera('./tensorflow-face-detection/media/test.mp4')

    frame = cam.get_frame()
    if out is None:
        [h, w] = frame.shape[:2]
        out = cv2.VideoWriter(args.OUTPUTDIR, 0, 25.0, (w, h))

    from face_detection.face_detection_opencv import FaceDetectorCV, draw_opencv
    detector_cv = FaceDetectorCV()
    from face_recognitions.face_recog_opencv import FaceRecogCV
    recog_cv = FaceRecogCV(args.FACEDIR)

    while True:
        frame = cam.get_frame()
        # frame = np.array(frame)
        # print(f"frame shape : {frame.shape}" )
        dt_result = detector_cv.DetectROI(frame)
        recog_cv.Recog(frame, detector_cv.face_locations, dt_result)
        draw_image = draw_opencv(frame.copy(),
                                 detector_cv.face_locations, dt_result.names,
                                 is_draw_text=True)

        # show the frame
        # cv2.imshow("Frame", frame)
        cv2.imshow("Frame", draw_image)
        out.write(draw_image)

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    print('finish')
