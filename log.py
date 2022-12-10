import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
save_name = "output.mp4"
fps = 30
width = 1280
height = 720
output_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2)))
        out.write(cv2.resize(image, output_size ))



        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()


    cv2.destroyAllWindows()