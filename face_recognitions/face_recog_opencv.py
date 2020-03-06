import numpy as np
import face_recognition
from common.FaceRecog import FaceRecog

class FaceRecogCV(FaceRecog):
    def __init__(self, file_path):
        super()._init(file_path)
        for face in self.facelist:
            face.embedding = face_recognition.face_encodings(face.image)
            print(f"face.embedding shape : {len(face.embedding[0])}")
            assert face.embedding != None
            face.embedding = face.embedding[0]
        print(f"self.facelist : {self.facelist}")

    def Recog(self, frame, face_locations, dt_result):
        rgb_small_frame = frame[:, :, ::-1]
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        known_face_encodings = []
        for value in self.facelist:
            known_face_encodings.append(value.embedding)
        known_face_encodings = np.array(known_face_encodings)

        for face_encoding in face_encodings:
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            min_value = min(distances)

            # tolerance: How much distance between faces to consider it a match. Lower is more strict.
            # 0.6 is typical best performance.
            name = "Unknown"
            if min_value < 0.6:
                index = np.argmin(distances)
                name = self.facelist[index].name
            dt_result.names.append(name)
            dt_result.classes.append(index + 1)

