# Realtime-facerecognition
Facerecognition using OpenCV and Dlib

Using Dlibs HOG detector as it was writen to be used on Rasberry Pi, using Dlib python wrapper face_recognition.
Writen using Linux Mint, all requered libraries where installed as root as the are well known.
This serverd as a base for MIPRO article, that code is in private repository as it contains personal photos.

To train the software, just create a folder with persons name in folder KnownPeople

example:
          KnownPeople/PersonA
          
And put at least 10 pictures of the person inside.

After the pictures are in folder, just runn StartProgram.py it will automatically create people vectors and start the camera feed.

Requered libs:

          sudo apt-get install build-essential cmake
          sudo apt-get install libopenblas-dev liblapack-dev
          sudo apt-get install libx11-dev libgtk-3-dev #Just for GUI
          sudo apt-get install python3 python3-dev python3-pip
          
          sudo -H pip3 install numpy
          sudo -H pip3 install imutils
          sudo -H pip3 install opencv-contrib-python #Version with proprietary algorithms  
          sudo -H pip3 install dlib
          sudo -H pip3 install face_recognition
         
For use on Rasberry, check stable version of OpenCV.

