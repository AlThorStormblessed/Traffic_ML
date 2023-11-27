import cv2
import numpy as np
import keras

vidcap = cv2.VideoCapture("Accident.mp4")

model = keras.models.load_model("Saved_model/model")

frame_width = int(vidcap.get(3))
frame_height = int(vidcap.get(4))
result = cv2.VideoWriter('Accident_detection.mp4', 
                         cv2.VideoWriter_fourcc(*'MP4V'),
                         24, (frame_width, frame_height))

success,image = vidcap.read()
count = 0

while success:
    cv2.imwrite("Traffic/Frame/frames/frame%d.jpg" % count, image)     # save frame as JPEG file      

    # image.resize((250,250))
    image_arr = np.array(image)
    image_arr.resize((1, 250, 250, 3))

    # success, image = vidcap.read()
    image_arr = np.array(image)
    image_arr.resize((1, 250, 250, 3))
    if(model.predict([[image_arr]])[0][0] > 0.8):
        prediction = "Accident"
    else:
        prediction = "Not Accident"
    
    cv2.putText(image, prediction, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0), 1)
    # print(count, ":", prediction)
    count += 1

    result.write(image)

    success,image = vidcap.read()

result.release()