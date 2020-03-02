import pickle
import cv2
import os
import face_recognition
import dlib
import numpy


def recognise_face_picture_test():
    path_to_encodings = os.getcwd() + "/KnownEmbeddings/embeddings.pickle"
    path_to_target_test_image = os.getcwd() + "/TestImage/1.png"

    # loads known encodings
    known_encodings = pickle.loads(open(path_to_encodings, "rb").read())

    # loads test image and converts it
    image = cv2.imread(path_to_target_test_image)
    converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Locates coordinates of faces and computes facial embeddings
    print("Finding faces")
    face_location_boxes = face_recognition.face_locations(converted_image, model="hog")
    encodings = face_recognition.face_encodings(converted_image, face_location_boxes)

    names_in_picture = []

    for encoding in encodings:
        # attempts to match faces in image to known faces
        matches = face_recognition.compare_faces(known_encodings["encodings"], encoding, tolerance= 0.00005)
        #probability = face_recognition.face_distance(known_encodings["encodings"], encoding)
        name = "Unknown"


        if True in matches:
            matched_ids = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            detector = dlib.get_frontal_face_detector()
            det, scores, idx = detector.run(image,1,-1)
            for i, d in enumerate(det):
                print("Detection {}, score: {} face_type{}".format( d, scores[i], idx[i]))
                per = scores[i]

            for i in matched_ids:
                name = known_encodings["names"][i]
                counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
                detector = dlib.get_frontal_face_detector()
                det, scores, idx = detector.run(image, 1, -1)
                for j, d in enumerate(det):
                    print("Detection {}, score: {} face_type{}".format(d, scores[j], idx[j]))
                    per = numpy.mean(scores[j])

            names_in_picture.append(name)
    text_picture = name + " " + str(per)
    for ((top, right, bottom, left), name) in zip(face_location_boxes, names_in_picture):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, text_picture, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("image", image)
    cv2.waitKey(0)

recognise_face_picture_test()