import cv2

threshold = 0.5   # to detect objects

cap = cv2.VideoCapture(0)
cap.set(3,648)
cap.set(4,488)

categories = []
catFile = 'object.names'
with open(catFile, 'rt') as f:
    categories = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, configs, boundBox = net.detect(img, confThreshold=threshold)
    print(classIds, boundBox)

    if len(classIds)!=0:
        for classId, confidence, box in zip(classIds.flatten(), configs.flatten(), boundBox):
            cv2.rectangle(img, box, color=(0,255,0), thickness=2)
            cv2.putText(img, categories[classId-1], (box[0]+5,box[1]+10), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,255,0), 1)
            cv2.putText(img, str(round(confidence*100, 2)), (box[0]+60,box[1]+10), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,255,0), 1)

    cv2.imshow('Output', img)
    cv2.waitKey(1)