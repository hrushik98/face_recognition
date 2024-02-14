import os
import face_recognition

def load_images_from_folder(folder):
    images = []
    labels = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                image = face_recognition.load_image_file(img_path)
                images.append(image)
                labels.append(subfolder)
    return images, labels

def train_face_recognition_model(dataset_folder):
    images, labels = load_images_from_folder(dataset_folder)

    face_encodings = [face_recognition.face_encodings(image)[0] for image in images]

    known_face_encodings = face_encodings
    known_face_labels = labels

    return known_face_encodings, known_face_labels

# Example usage:
dataset_folder = "/home/phani/Desktop/facerecog/known_faces"
known_face_encodings, known_face_labels = train_face_recognition_model(dataset_folder)

import face_recognition
import cv2

def recognize_faces(known_face_encodings, known_face_labels, test_image_path):
    test_image = face_recognition.load_image_file(test_image_path)
    face_locations = face_recognition.face_locations(test_image)
    face_encodings = face_recognition.face_encodings(test_image, face_locations)

    recognized_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_labels[first_match_index]

        recognized_names.append(name)

    return recognized_names

# Example usage:
test_image_path = "elon-unknown.jpeg"
result = recognize_faces(known_face_encodings, known_face_labels, test_image_path)
print(result)
