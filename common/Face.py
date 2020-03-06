import os
import cv2
class Face:
    def __init__(self):
        self.filepath = None
        self.name = None
        self.image = None
        self.id = None
        self.embedding = None

    @staticmethod
    def readFaceDir(face_file_path):
        assert os.path.isdir(face_file_path)
        face_file_list = os.listdir(face_file_path)

        face_list = []
        id_count = 0
        for file_image in face_file_list:
            name, ext =  os.path.splitext(file_image)
            if ext == '.jpg' or ext == '.png':
                face = Face()
                face.filepath = os.path.join(face_file_path, file_image)
                face.image = cv2.imread(face.filepath)
                face.image
                face.image = face.image[:, :, ::-1] # bgr -> rgb
                face.name = name
                face.id = id_count
                id_count += 1
                face_list.append(face)
        return face_list
