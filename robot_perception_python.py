import cv2
import PIL
import random
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.models.detection import maskrcnn_resnet50_fpn

#this loads the the model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#class names of COCO are defined
coco_names = ['unlabeled', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in coco_names]

#start the webcam
cap = cv2.VideoCapture(0)

#Mask R-CNN images a re preporcessed
def process(image):
    image = PIL.Image.fromarray(image)
    tensor_image = torchvision.transforms.functional.to_tensor(image).to(device)
    return tensor_image.unsqueeze(0)

#center point of segmented mask is calculated
def center_mask(mask):
    pixels = np.nonzero(mask)
    x = np.mean(pixels[1])
    y = np.mean(pixels[0])
    return int(x), int(y)

#extracts principal axes using PCA
def get_axes(mask):
    zs = mask.shape
    data_pts = np.zeros((zs[0] * zs[1], 2), dtype=np.float64)
    for i in range(zs[0]):
        for j in range(zs[1]):
            data_pts[i * zs[1] + j, 0] = j
            data_pts[i * zs[1] + j, 1] = i
    #PCA analysis is perfomed
    pca = cv2.PCACompute2(data_pts, mean=None)
    eigen = pca[1]
    return eigen

def visualtization(image, output):
    result = image.copy()
    for box, label, score, mask in zip(output['boxes'], output['labels'], output['scores'], output['masks']):
        if score > 0.5:
            color = colors[label]
            # Draw bounding box
            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(result, c1, c2, color, thickness=2)
            # Draw text
            display_txt = "{}: {:.1f}%".format(coco_names[label], 100*score)
            cv2.putText(result, display_txt, (c1[0], c1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Convert mask to numpy array
            mask = mask[0].mul(255).byte().cpu().numpy()
            # Calculate center point
            center = center_mask(mask)
            # Draw center point
            cv2.circle(result, center, 3, color, -1)
            
            # Extract principal axes
            axes = get_axes(mask)
            # Draw principal axes
            pa1 = (int(center[0] + axes[0][0] * 50), int(center[1] + axes[0][1] * 50))
            pa2 = (int(center[0] + axes[1][0] * 50), int(center[1] + axes[1][1] * 50))
            cv2.arrowedLine(result, center, pa1, color, 2)
            cv2.arrowedLine(result, center, pa2, color, 2)     
    return result

#main loop that loads everything
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    #process the frame and performs the  inference
    tensor_image = process(frame)
    with torch.no_grad():
        output = model(tensor_image)[0]

    frame_result = visualtization(frame, output)

    #final result is displayed on webcam
    cv2.imshow('Mask R-CNN', frame_result)
    
    #if user presses key 1 or q then it stops the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#webcam is stopped and window is deleted
cap.release()
cv2.destroyAllWindows()
