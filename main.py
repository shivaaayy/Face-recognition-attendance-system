import csv 
import cv2
import numpy as np
import face_recognition
import os
from datetime import *

#Loading images using path
path = 'E:/PROJECT/FINAL_CSV/images'
images = []
names = []
List = os.listdir(path)
#print(List)

for name in List:
    img = cv2.imread(f'{path}/{name}')
    images.append(img)
    names.append(os.path.splitext(name)[0])

#print(names)
#Function to find encodings for all images in directory
def encode(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeImg = face_recognition.face_encodings(img)[0]
        encode_list.append(encodeImg)
    return encode_list

#function for attendance.csv file which records with the dedicated time in the python file
def record_attendance(name):
    now = datetime.now()
    dt = now.strftime('%H:%M:%S')
    with open('attendance.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if name == row[1]:
                # The person has already been recorded today, so do not update the time
                return

    # # The person has not been recorded today, so record & the time
    file = open('attendance.csv', 'a', newline='')
    writer = csv.writer(file)
    writer.writerow([name, dt])
    file.close()

print("Encoding Images...")
encodeList = encode(images)
print("Encoding Completed.")

#starting webcam
cap = cv2.VideoCapture(0)

#in this loop the image size is reduced to 1/4th as it helps in fast rendering and improve the performance
while True:
    success, img = cap.read()
    #Reducing size of real-time image to 1/4th
    imgResize = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgResize = cv2.cvtColor(imgResize, cv2.COLOR_BGR2RGB)

    # Finding face in current frame
    face = face_recognition.face_locations(imgResize)
    # Encode detected face
    encodeImg = face_recognition.face_encodings(imgResize, face)

    #Finding matches with existing images
    for encodecurr, loc in zip(encodeImg, face):
        match = face_recognition.compare_faces(encodeList, encodecurr)
        faceDist = face_recognition.face_distance(encodeList, encodecurr)
        print(faceDist)
        #Lowest or nearest distance will be best match
        index_BestMatch = np.argmin(faceDist)

        if match[index_BestMatch]:
            name = names[index_BestMatch]
            y1,x2,y2,x1 = loc
            #Retaining original image size for rectangle location
            y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)
            cv2.rectangle(img,(x1,y2-30),(x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+8, y2-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255),2)
            record_attendance(name)

    #this is to check the current date and time for the csv file
    current_time = datetime.now()
    csv_file_name = f'attendance{current_time.strftime("%Y-%m-%d")}.csv'
    yesterday = current_time - timedelta(days=1)

    #this checks that after 24hrs a new csv file is created with date mentioned on it
    if current_time.day != yesterday.day:
        with open('attendance.csv', 'r') as f:
            with open(f'attendance {current_time.strftime("%d-%m-%Y")}.csv', 'w', newline='') as csvfile:
                reader = csv.reader(f, delimiter=',')
                writer = csv.writer(csvfile, delimiter=',')
                for row in reader:
                    writer.writerow(row)
        os.remove('attendance.csv')
        
        #this is to check if the previous attendance file exists or not 
        #if exists then it will store the data in it and if not then it will create a new one 
        if not os.path.exists('attendance.csv'):
            with open('attendance.csv', 'w', newline='') as csvfile:
                fieldnames = ['emp_id', 'name', 'time']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

    #this is for showing the live feed 
    cv2.imshow('Webcam', img)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break