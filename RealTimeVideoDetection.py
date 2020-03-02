import os
import pickle
import time
import imutils
from imutils.video import VideoStream
import cv2
import dlib
import face_recognition
import FaceDistanceToConfidence
import numpy

def real_time_detection():
    path_to_encodings = os.getcwd() + "/KnownEmbeddings/embeddings.pickle"

    print("Loading encodings")
    known_people = pickle.loads(open(path_to_encodings, "rb").read())

    req_prob = 75.0
    floor_detection_threshold = 35.0
    print("Starting video stream")
    video_stream = VideoStream(src=0).start()
    writer = None
    time.sleep(2.0)
    while True:
        frame = video_stream.read()
        frame_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # if working on Rasberyy change width=320 and height=240 to spead up 
        frame_image = imutils.resize(frame, width=640, height=480)
        r = frame.shape[1] / float(frame_image.shape[1])
        cv2.imshow("Frame", frame)

        names_in_picture = []
        perc_in_picture = []
        # locates face in the converted image, up-sampling set to 0 and model HOG selected
        faces_locations_boxes = face_recognition.face_locations(frame_image, number_of_times_to_upsample=0, model="hog")
        # creates encodings of found faces
        found_faces_encodings = face_recognition.face_encodings(frame_image, faces_locations_boxes)

        for encoding in found_faces_encodings:
            # looks in our dictionary of known people to see if we have a match, tolerance is set as stated
            # in dlib's documentation, 0.6 - person is the same, lower makes better detection
            for person_name, person_encoding in known_people.items():
                fd = face_recognition.face_distance(person_encoding, encoding)
                confidence = FaceDistanceToConfidence.face_distance_to_confidence(fd)
                percentage = numpy.mean(confidence) * 100

                if percentage > floor_detection_threshold:
                    # Needs to be replaced with a dictionary
                    names_in_picture.append(person_name)
                    perc_in_picture.append(percentage)

                    # This was for the unlock part of the work, need to update this part to last version
                    # if percentage > req_prob:
                        # print("Send signal to unlock")

        for ((top, right, bottom, left), person_name, person_percentage) in zip(faces_locations_boxes, names_in_picture, perc_in_picture):
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)

            cv2.rectangle(frame, (top, left), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            text_to_picture = "{}: {:.2f}%".format(person_name, person_percentage)
            cv2.putText(frame, text_to_picture, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            # This part was a attempt to ease the load on Rasberry
            # time.sleep(0.3)
            # names_in_picture.clear()
            # perc_in_picture.clear()
            # time.sleep(0.3)


        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            video_stream.stop()
            break

