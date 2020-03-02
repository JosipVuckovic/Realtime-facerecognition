from imutils import paths
import os
import pickle
import cv2
import face_recognition


def extract_known_embeddings():
    path_to_images = os.getcwd() + "/KnownPeople/"
    path_to_encodings = os.getcwd() + "/KnownEmbeddings/embeddings.pickle"
    image_paths = list(paths.list_images(path_to_images))


    known_people = {}

    for (i, image_path) in enumerate(image_paths):
        # gets persons name from the folder
        print("Processing image {}/{}".format(i + 1, len(image_paths)))
        person_name = image_path.split(os.path.sep)[-2]

        # loads input image and converts it to dlib format
        converted_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # finds the face coordinates in the image and applies HOG model
        face_location_boxes = face_recognition.face_locations(converted_image, model="hog")
        # computes the embedding for the face
        encodings = face_recognition.face_encodings(converted_image, face_location_boxes)

        # loops over encodings and adds adds name and encoding to our set
        for encoding in encodings:
            known_people[person_name] = [encoding]


    # saves encodings to KnownEmbeddings folder
    print("Saving encodings")
    #encoding_data = {"encodings": known_people_encodings, "names": known_people_names}
    encoding_data = known_people
    file = open(path_to_encodings, "wb")
    file.write(pickle.dumps(encoding_data))
    file.close()

