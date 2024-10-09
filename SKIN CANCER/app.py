import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the model
model = load_model('my_cnn_model.h5')

#yeni gelen resmi modelin girdi boyutuna uygun hale getirelim 
def process_image(image): 
    image = image.resize((170,170)) 
    image = np.array(image) 
    image = image / 255.0  
    image = np.expand_dims(image, axis=0) # burada modelin beklediği gibi bir girdi oluşturduk
    return image


st.title("Skin Cancer Classification - Metehan Ayhan")
st.write("This is a simple image classification web app to predict the type of skin cancer.")
st.write("Please upload a skin image for the prediction.")

file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])

if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file) # resmi aç
    st.image(image, use_column_width=True, caption='Image:') # resmi gösterelim
    predictions = model.predict(process_image(image))
    predicted_class = np.argmax(predictions) # en yüksek olasılığa sahip sınıfı al
    
    class_names = ['Cancer', 'Not Cancer']

    st.write(class_names[predicted_class], "with", round(100*np.max(predictions), 2), "% probability")
