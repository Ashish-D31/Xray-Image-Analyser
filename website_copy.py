import streamlit as st
import os
import cv2
import numpy as np
import tensorflow as tf

# Load TensorFlow Lite models
XrayOrNot_interpreter = tf.lite.Interpreter(model_path="XrayOrNot.tflite")
FracOrNot_interpreter = tf.lite.Interpreter(model_path="FracOrNo.tflite")

# Allocate tensors
XrayOrNot_interpreter.allocate_tensors()
FracOrNot_interpreter.allocate_tensors()

# Get input and output details
XrayOrNot_input_details = XrayOrNot_interpreter.get_input_details()
XrayOrNot_output_details = XrayOrNot_interpreter.get_output_details()

FracOrNot_input_details = FracOrNot_interpreter.get_input_details()
FracOrNot_output_details = FracOrNot_interpreter.get_output_details()

# Function to make predictions
def predict_image(img_path):
    resize_dims = (128, 128)
    img = cv2.imread(img_path)
    img = cv2.resize(img, resize_dims)
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # Set input tensors
    XrayOrNot_interpreter.set_tensor(XrayOrNot_input_details[0]['index'], img)
    FracOrNot_interpreter.set_tensor(FracOrNot_input_details[0]['index'], img)

    # Run inference
    XrayOrNot_interpreter.invoke()
    FracOrNot_interpreter.invoke()

    # Get output tensors
    Xray_pred = XrayOrNot_interpreter.get_tensor(XrayOrNot_output_details[0]['index'])[0][0] * 100
    Frac_pred = FracOrNot_interpreter.get_tensor(FracOrNot_output_details[0]['index'])[0][0] * 100

    Xray_pred = round(Xray_pred, 3)
    Frac_pred = round(Frac_pred, 3)

    return Frac_pred, Xray_pred

# Streamlit app
def main():
    st.title("Bone Fracture Detection")

    uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        # Save uploaded file temporarily
        file_path = os.path.join("temp_dir", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display the uploaded image
        display_img = cv2.imread(file_path)
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        st.image(display_img, caption='Uploaded Image', use_column_width=True)

        # Make predictions
        Frac_pred, Xray_pred = predict_image(file_path)

        # Interpret predictions
        if Xray_pred < 70:
            st.write("Image is not an X-ray. Please upload an X-ray image.")
        else:
            if Frac_pred < 70:
                st.write("The X-ray is not of a fractured bone.")
                st.write(f"X-ray Prediction Confidence: {Xray_pred}%")
                st.write(f"Fracture Prediction Confidence: {Frac_pred}%")
            else:
                st.write("The X-ray is of a fractured bone.")
                st.write(f"Confidence: {(100 - Xray_pred).round(3)}%")

if __name__ == "__main__":
    if not os.path.exists("temp_dir"):
        os.makedirs("temp_dir")
    main()
