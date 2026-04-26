import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="Potato Disease Scanner", page_icon="🥔")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # We use the full tf.lite here because it supports newer opcodes
    interpreter = tf.lite.Interpreter(model_path="potato_model.tflite")
    interpreter.allocate_tensors()
    with open('class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
    return interpreter, class_names

interpreter, class_names = load_assets()

def predict(img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    img_array = img_array.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# --- UI ---
st.title("🥔 Potato Leaf Disease Detection")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf Image', use_container_width=True)
    
    img_resized = image.resize((256, 256))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    result = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.divider()
    st.success(f"### Result: {result}")
    st.write(f"**Confidence:** {confidence:.2f}%")
