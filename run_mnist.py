import cv2
import numpy as np
from tensorflow.keras.models import load_model

cap = cv2.VideoCapture(0)
WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

model = load_model('models/mnist_model.h5')


def prediction(image, model):
    img = cv2.resize(image, (28, 28))
    img = img / 255
    img = img.reshape(1, 28, 28, 1)
    predict = model.predict(img)
    prob = np.amax(predict)
    result = np.argmax(predict)
    if prob < 0.75:
        result = 0
        prob = 0
    return result, prob


while True:

    _, frame = cap.read()
    frame_copy = frame.copy()

    bbox_size = (120, 120)
    bbox = [(int(WIDTH // 2 - bbox_size[0] // 2), int(HEIGHT // 2 - bbox_size[1] // 2)),
            (int(WIDTH // 2 + bbox_size[0] // 2), int(HEIGHT // 2 + bbox_size[1] // 2))]

    img_cropped = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 3, 2)
    thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    dilated = cv2.dilate(thresh, kernel)

    dilated = cv2.resize(dilated, (200, 200))
    cv2.imshow("cropped", dilated)

    result, prob = prediction(dilated, model)
    # put text on the frame showing the prediction and probability at top left of screen
    cv2.putText(frame_copy, f'Numero Detectado: {result}', (
        10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
    cv2.putText(frame_copy, f'Certeza: {prob:.2f}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)

    if prob > 0.75:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    cv2.rectangle(frame_copy, bbox[0], bbox[1], color, 2)
    cv2.imshow("input", frame_copy)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
