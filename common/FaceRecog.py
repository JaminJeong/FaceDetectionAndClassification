from common.Face import Face

class FaceRecog:
    def _init(self, face_path):
        self.facelist = Face.readFaceDir(face_path)

    def Recog(self, frame):
        pass
