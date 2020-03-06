class FaceDetector:
    class FaceDetectedResult():
        def __init__(self):
            self.scores = []
            self.boxes = []
            self.names = []
            self.classes = []

    def _init(self):
        self.faceDetectedResult = FaceDetector.FaceDetectedResult()

    def DetectROI(self, frame):
        return self.faceDetectedResult
