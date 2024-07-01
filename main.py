import cv2
import numpy as np
from PIL import Image
import streamlit as st
MODEL = r"./model/MobileNetSSD_deploy.caffemodel"
PROTOTXT = r"./model/MobileNetSSD_deploy.prototxt.txt"


def process_image(image):
    blob = cv2.dnn.blobFromImage(cv2.resize(
        image, (300, 300)), 0.007843, (300, 300), 127.5)
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    net.setInput(blob)
    return net.forward()


def annotate_image(image, detections, confidence_threshold=0.5):
    # loop over the detections
    (h, w) = image.shape[:2]

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startx, starty, endx, endy) = box.astype("int")
            cv2.rectangle(image, (startx, starty), (endx, endy), 70, 2)
    return image


def main():
    st.title('Object Detection for Images')
    file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])
    # Create a sidebar for the vertical slider
    threshold = st.sidebar.slider(
        "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.02)

    # check validity of uploaded file
    if file is not None:
        st.image(file, caption='Uploaded Image')    # show the input image
        image = np.array(Image.open(file))
        detections = process_image(image)
        processed_image = annotate_image(
            image, detections, threshold)  # output image
        st.image(processed_image, caption='Processed Image')    # show output image


if __name__ == "__main__":
    main()
