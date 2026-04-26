import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="Potato Disease Scanner", page_icon="🥔")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path="potato_model.tflite")
    interpreter.allocate_tensors()
    
    with open('class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
    return interpreter, class_names

interpreter, class_names = load_assets()

# --- HELPER FUNCTION FOR TFLITE PREDICTION ---
def predict_tflite(img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Ensure input is float32 (standard for TFLite)
    img_array = img_array.astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# --- UI ---
st.title("🥔 Potato Leaf Disease Detection")
st.write("Upload a photo of a potato leaf to identify if it is healthy or diseased.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf Image', use_container_width=True)
    
    # Preprocessing
    st.write("Analyzing...")
    img_resized = image.resize((256, 256))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    # Prediction using TFLite logic
    predictions = predict_tflite(img_array)
    
    # Apply softmax to get probabilities
    score = tf.nn.softmax(predictions[0])
    
    result = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    # Display Result
    st.divider()
    if "healthy" in result.lower():
        st.success(f"### Result: {result}")
    else:
        st.error(f"### Result: {result}")
        
    st.write(f"**Confidence Level:** {confidence:.2f}%")
    
    if "Early_blight" in result:
        st.info("Advice: Apply fungicides and ensure proper crop rotation.")
    elif "Late_blight" in result:
        st.warning("Advice: High risk! Remove infected plants and apply late-blight specific treatments.")
