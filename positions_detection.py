import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from modelscope.utils.cv.image_utils import draw_face_detection_result
from modelscope.preprocessors.image import LoadImage
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

window_size = 80
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,  
                                  max_num_faces=1,         
                                  refine_landmarks=True,  
                                  min_detection_confidence=0.5, 
                                  min_tracking_confidence=0.5)

eye_landmarks = [159, 145, 33, 133, 386, 374, 362, 263]  # top down left right

def roi_detection(image_path):
    image = cv2.imread(image_path)
    eyes = face_mesh.process(image)
    try: 
        face = eyes.multi_face_landmarks[0]
    except TypeError:
        print(f"error: {image_path}")

    # get the positions of the eyes
    x1_min = int(face.landmark[33].x * image.shape[1] - window_size)
    y1_min = int(face.landmark[159].y * image.shape[0] - window_size)
    x1_max = int(face.landmark[133].x * image.shape[1] + window_size)
    y1_max = int(face.landmark[145].y * image.shape[0] + window_size)

    x2_min = int(face.landmark[362].x * image.shape[1] - window_size)
    y2_min = int(face.landmark[386].y * image.shape[0] - window_size)
    x2_max = int(face.landmark[263].x * image.shape[1] + window_size)
    y2_max = int(face.landmark[374].y * image.shape[0] + window_size)

    return [y1_min, y1_max, x1_min, x1_max], [y2_min, y2_max, x2_min, x2_max]

def eye_detection(roi, positions):
  rows, cols, _ = roi.shape

  gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (1,1), 0)  
  _, threshold = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY_INV)
  kernel = np.ones((7,7),np.uint8)
  closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations = 2)

  contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

  (x, y, w, h) = (0, 0, 0, 0) 

  for cnt in contours:
      (x,y),radius = cv2.minEnclosingCircle(cnt)
      # get the center (x, y) and radius of the circle
      diameter = radius * 2
      
      if diameter > 100:
          continue
          
      (x, y, w, h) = cv2.boundingRect(cnt)

      #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
      # cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
      # cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
      # cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
      break
  
  positions.append([x + w/2,  y + h/2])
  return positions
  

def crop_square(img, second_coord):
    y1, y2, x1, x2 = second_coord

    H, W = img.shape[:2]
    if x2 > W:
        x2 = W
    if y2 > H:
        y2 = H 
    h = y2 - y1
    w = x2 - x1
    side = np.maximum(h, w)
    y = y1 + (h - side) // 2
    x = x1 + (w - side) // 2

    if x < 0:
        x = 0    
    if y < 0:
        y = 0
    if x + side > W:
        side = W - x
    if y + side > H:
        side = H - y

    return img[y:y+side, x:x+side]    

def process_eyes(big_folder, save_folder):
    dict_l = {}
    dict_r = {}
    iterator = tqdm(os.listdir(big_folder))
    
    for idx, subfolder in enumerate(iterator):

        subfolder_path = os.path.join(big_folder, subfolder)
        image_names = os.listdir(subfolder_path)
        leye_positions = []
        reye_positions = []
        
        l_eye_coo, r_eye_coo  = roi_detection(os.path.join(subfolder_path, f'{len(image_names)//2}.jpg'))
        for idx in range(0, len(image_names)):
            image_path = os.path.join(subfolder_path, f'{idx}.jpg')
            image = cv2.imread(image_path)

            leye = crop_square(image, l_eye_coo)
            reye = crop_square(image, r_eye_coo)

            leye_positions = eye_detection(leye, leye_positions)
            reye_positions = eye_detection(reye, reye_positions)
        
        dict_l[subfolder] = leye_positions
        dict_r[subfolder] = reye_positions
        # print('l:', leye_positions)
        # print('r:', reye_positions)

    with open(os.path.join(save_folder, 'leye_positions.pkl'), 'wb') as f:
        pickle.dump(dict_l, f)

    with open(os.path.join(save_folder, 'reye_positions.pkl'), 'wb') as f:
        pickle.dump(dict_r, f)


def process_nose(big_folder, save_folder):
    dict_nose = {}
    face_detection1 = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface')
    face_detection2 = pipeline(task=Tasks.face_detection, model='damo/cv_resnet_facedetection_scrfd10gkps')
    iterator = tqdm(os.listdir(big_folder))
  
    for subfolder in iterator:

        subfolder_path = os.path.join(big_folder, subfolder)
        image_names = os.listdir(subfolder_path)
        nose_positions = []
        for idx in range(0, len(image_names)):
            image_path = os.path.join(subfolder_path, f'{idx}.jpg')
            result1 = face_detection1(image_path)
            result2 = face_detection2(image_path)
            x1 = result1['keypoints'][0][4]
            y1 = result1['keypoints'][0][5]
            x2 = result2['keypoints'][0][4]
            y2 = result2['keypoints'][0][5]
            x = (x1 + x2)/2
            y = (y1 + y2)/2
            nose_positions.append([x, y])
        dict_nose[subfolder] = nose_positions
    with open(os.path.join(save_folder, 'nose_positions.pkl'), 'wb') as f:
        pickle.dump(dict_nose, f)

   
if __name__ == '__main__':
    
    data_path = "/home/ivi/ivi/oyy/vHit/data/dataset_raw"
    save_path = f"/home/ivi/ivi/oyy/vHit/data/positions{window_size}ws"
    process_eyes(data_path, save_path)
    process_nose(data_path, save_path)
    test_data_path = "/home/ivi/ivi/oyy/vHit/data/dataset_test"
    test_save_path = f"/home/ivi/ivi/oyy/vHit/data/positions{window_size}ws_test"
    process_eyes(test_data_path, test_save_path)
    process_nose(test_data_path, test_save_path)