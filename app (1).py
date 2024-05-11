from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import face_recognition
import os
import tempfile
import os

app = FastAPI()

directory = os.path.join('server', 'assets', 'imgs')


def get_image_paths(directory):
    image_paths = []
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.png', '.JPG')): 
            image_path = os.path.join(directory, filename)
            image_paths.append(image_path)
    return image_paths


def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def encodeStudent(image):
    print('encoding user')
    encodedUser = faceEncodings([image])[0]
    return encodedUser

def findStudent(encodedUser, imagePaths):
    print('inside the finding user')
    for imagePath in imagePaths:
        try:
            frame = cv2.imread(imagePath)
            if frame is None:
                print(f"Failed to load image: {imagePath}")
                continue
            faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            print('frame is set')

            facesCurrentFrame = face_recognition.face_locations(faces)
            encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

            for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
                print('matching face')
                matches = face_recognition.compare_faces([encodedUser], encodeFace)
                faceDis = face_recognition.face_distance([encodedUser], encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    return 1 
        except Exception as e:
            print(f"Error processing image {imagePath}: {e}")

    return 0


def count_faces_in_image(image, imagePaths):
    input_image = cv2.imread(image)
    input_faces = cv2.resize(input_image, (0, 0), None, 0.25, 0.25)
    input_face_locations = face_recognition.face_locations(input_faces)
    input_face_encodings = face_recognition.face_encodings(input_faces, input_face_locations)

    total_faces_found = 0
    matched_images = [] 

    for imagePath in imagePaths:
        frame = cv2.imread(imagePath)
        if frame is None:
            print(f"Failed to load image: {imagePath}")
            continue
        faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)

        face_locations = face_recognition.face_locations(faces)
        face_encodings = face_recognition.face_encodings(faces, face_locations)

        for encodeFace in face_encodings:
            matches = face_recognition.compare_faces(input_face_encodings, encodeFace)
            if True in matches:
                total_faces_found += 1
                matched_images.append(imagePath) 
    path_components = []
    for image_path in matched_images:
        path_components.append(image_path.split(os.sep)[3] )
    print(total_faces_found, path_components )
    return total_faces_found, path_components 

@app.post("/mark_attendance")
async def mark_attendace(image: UploadFile = File(...)):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(await image.read())
        temp_file.close()

        print(f"Temporary file created: {temp_file.name}")

        testImage = encodeStudent(cv2.imread(temp_file.name))
        print("Image encoding completed.")
        image_paths = get_image_paths(directory)
        rsult = findStudent(testImage, imagePaths=image_paths)
        print("Student finding completed.")

        os.unlink(temp_file.name)
        print("Temporary file deleted.")

        return {"message": "Success", "attendance": rsult}
    except Exception as e:
        print(f"Error processing request: {e}")
        return {"message": "Error", "error": str(e)}

# @app.post("/check_student")
# async def check_stuednt(image: UploadFile = File(...)):
#     temp_file = tempfile.NamedTemporaryFile(delete=False)
#     temp_file.write(await image.read())
#     temp_file.close()
#     image_paths = get_image_paths(directory)
#     rslt, stds = count_faces_in_image('t.JPG', imagePaths=image_paths)
#     return {"message": "Success" , "total":rslt, "students":stds}


# @app.get("/check")
# async def check_stuednt():
    
#     image_paths = get_image_paths(directory)
#     print(image_paths)
#     rslt, stds = count_faces_in_image('t.JPG', imagePaths=image_paths)
#     return {"message": "Success" , "total":rslt, "students":stds}


