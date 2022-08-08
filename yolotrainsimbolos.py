import cv2
import numpy as np
import torch
from playsound import playsound
from yolov5 import YOLOv5
from flask import Flask, flash
torch.cuda.empty_cache()

device = "cpu" # or "gpu
model_path = "C:/Users/Alexandre/yolov5/runs/train/exp2/weights/best.pt"
yolov5 = YOLOv5(model_path, device)
a = 0
app = Flask(__name__)
app.secret_key = "secret key"
dict_Results = {
    0: ('Símbolo Detetado: Lavar de forma Manual', "C:/Users/Alexandre/PycharmProjects/pythonProject/pythonProject/audio/LavagemManual.mp3"),
    1: ('Símbolo Detetado: Lavar até 30 graus', "C:/Users/Alexandre/PycharmProjects/pythonProject/pythonProject/audio/Lavar30graus.mp3"),
    2: ('Símbolo Detetado: Lavar até 40 graus', "C:/Users/Alexandre/PycharmProjects/pythonProject/pythonProject/audio/Lavar40graus.mp3"),
    3: ('Símbolo Detetado: Lavar até 50 graus', "C:/Users/Alexandre/PycharmProjects/pythonProject/pythonProject/audio/Lavar50graus.mp3"),
    4: ('Símbolo Detetado: Lavar até 60 graus', "C:/Users/Alexandre/PycharmProjects/pythonProject/pythonProject/audio/Lavar60graus.mp3"),
    5: ('Símbolo Detetado: Lavar até 70 graus', "C:/Users/Alexandre/PycharmProjects/pythonProject/pythonProject/audio/Lavar70graus.mp3"),
    6: ('Símbolo Detetado: Lavar até 95 graus', "C:/Users/Alexandre/PycharmProjects/pythonProject/pythonProject/audio/Lavar95graus.mp3"),
    7: ('Símbolo Detetado: Limpar a seco', "C:/Users/Alexandre/PycharmProjects/pythonProject/pythonProject/audio/Limparaseco.mp3"),
    8: ('Símbolo Detetado: Passar a vapor', "C:/Users/Alexandre/PycharmProjects/pythonProject/pythonProject/audio/Passaravapor.mp3"),
    9: ('Símbolo Detetado: Passar a 110 graus', "C:/Users/Alexandre/PycharmProjects/pythonProject/pythonProject/audio/Passar110graus.mp3"),
    10: ('Símbolo Detetado: Passar sem Vapor', "C:/Users/Alexandre/PycharmProjects/pythonProject/pythonProject/audio/Passarsemvapor.mp3"),
    11: ('Símbolo Detetado: Passar', "C:/Users/Alexandre/PycharmProjects/pythonProject/pythonProject/audio/Passar.mp3"),
    12: ('Símbolo Detetado: Proibido alvejar', "C:/Users/Alexandre/PycharmProjects/pythonProject/pythonProject/audio/Proibidoalvejar.mp3"),
    13: ('Símbolo Detetado: Proibido limpar a seco', "C:/Users/Alexandre/PycharmProjects/pythonProject/pythonProject/audio/Proibidolimparseco.mp3"),
    14: ('Símbolo Detetado: Proibido passar', "C:/Users/Alexandre/PycharmProjects/pythonProject/pythonProject/audio/Proibidopassar.mp3"),
    15: ('Símbolo Detetado: Proibido secar em tambor', "C:/Users/Alexandre/PycharmProjects/pythonProject/pythonProject/audio/Proibidosecartambor.mp3"),
    16: ('Símbolo Detetado: Proibido secar', "C:/Users/Alexandre/PycharmProjects/pythonProject/pythonProject/audio/Proibidosecar.mp3"),
    17: ('Símbolo Detetado: Secar na horizontal', "C:/Users/Alexandre/PycharmProjects/pythonProject/pythonProject/audio/Secarhorizontal.mp3")
}
def display_img(img):
    results = yolov5.predict(img)
    predictions = results.pred[0]
    categories = predictions[:, 5]
    results = np.squeeze(results.render())
    cv2.imwrite('static/detect/' + str(a) + '.jpg', results)
    if len(predictions) != 0:
        cat = list(categories)[0].detach().cpu().numpy()
        return cat


def test_img(img):
    results = yolov5.predict(img)
    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4]  # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]
    # show detection bounding boxes on image
    im = np.squeeze(results.render())
    info = results.pandas().xyxy[0].to_dict(orient="records")
    if len(info) != 0:
        playsound("C:/Users/Alexandre/PycharmProjects/pythonProject/pythonProject/audio/simbolosdetetados.mp3")
    for i in range(len(results.pandas())):
        info = results.pandas().xyxy[i].to_dict(orient="records")
        for result in info:
            information = result['class']
            if information != None and information >= 0 and information < len(dict_Results):
                flash(dict_Results[int(information)][0])
                playsound(dict_Results[int(information)][1])
    # #cv2.imshow("Predict Etiqueta", im)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()





