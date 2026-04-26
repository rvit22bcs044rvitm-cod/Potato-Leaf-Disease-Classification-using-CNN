import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

st.set_page_config(page_title="Potato Disease Scanner", page_icon="🥔")

@st.cache_resource
def load_assets():
    # Loading using the latest TF 2.17+ logic
    interpreter = tf.lite.Interpreter(model_path="potato_model.tflite")
    interpreter.allocate_tensors()
    with open('class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
    return interpreter, class_names

try:
    interpreter, class_names = load_assets()
    
    def predict(img_array):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        img_array = img_array.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])

    st.title("🥔 Potato Leaf Disease Detection")
    uploaded_file = st.file_uploader("Upload leaf photo", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
        # Preprocessing
        img_resized = image.resize((256, 256))
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        result = class_names[np.argmax(score)]
        st.success(f"### Diagnosis: {result}")
        st.write(f"**Confidence:** {100 * np.max(score):.2f}%")

except Exception as e:
    st.error(f"Deployment Sync Error: {e}")
    st.info("The server is still updating the AI engine. Please wait 2 minutes and refresh.")
