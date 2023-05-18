import tensorflow as tf
import numpy as np
import cv2
from process_labels import generate_labels

np.set_printoptions(suppress=True)
capture = cv2.VideoCapture(0)
loaded_model = tf.keras.models.load_model("C://Users//Bogdan//PycharmProjects//pythonProject//model.savedmodel")

input_data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
label_dict = generate_labels()

while True:
    font_style = cv2.FONT_HERSHEY_COMPLEX
    ret_val, video_frame = capture.read()
    video_frame = cv2.flip(video_frame, 1)
    if not ret_val:
        continue

    video_frame = cv2.rectangle(video_frame, (140, 40), (490, 440), (0, 255, 0), 3)
    cropped_frame = video_frame[80:360, 220:530]
    resized_frame = cv2.resize(cropped_frame, (224, 224))
    frame_array = np.asarray(resized_frame)
    norm_frame_array = (frame_array.astype(np.float32) / 127.0) - 1
    input_data[0] = norm_frame_array
    predictions = loaded_model.predict(input_data)
    result_idx = np.argmax(predictions[0])

    cv2.putText(video_frame, label_dict[str(result_idx)], (265, 467), font_style, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow("Object Detector", video_frame)

capture.release()
cv2.destroyAllWindows()
