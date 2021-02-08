import cv2
import numpy as np

limit = 0.45  # sensibilidade
algo = 0.2  # algoritmo para quebrar repetição da detecção
cam = cv2.VideoCapture(0)  # seleção de camera - se só tem uma no sistema, usar "0"

classNames = []
classFile = 'coco.names'

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# base com nomes
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

config = cv2.dnn_DetectionModel(weightsPath, configPath)
config.setInputSize(360, 360)
config.setInputScale(1.0 / 127.5)
config.setInputMean((127.5, 127.5, 127.5))
config.setInputSwapRB(True)

while True:
    success, img = cam.read()
    classIds, perga, bbox = config.detect(img, confThreshold=limit)
    bbox = list(bbox)
    perga = list(np.array(perga).reshape(1, -1)[0])
    perga = list(map(float, perga))

    indices = cv2.dnn.NMSBoxes(bbox, perga, limit, algo)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, h + y), color=(255, 255, 0), thickness=2)  # cor e grossura da identifição
        cv2.putText(img, classNames[classIds[i][0] - 1].upper(), (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sexta-Feira", img)  # nome da janela de saida de img
    key = cv2.waitKey(1)  # intervalo na tecla

    if key == 27:  # tecla esc pra sair
        break
