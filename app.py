import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib 
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# title
st.title("To'plarni klassifikatsiya qiluvchi model")

#rasm yuklash
file = st.file_uploader("Rasm yuklash", type=['png', 'gif', 'jpeg', 'jpg', 'svg'])


if file:
    st.image(file)

    #PIL convert
    img = PILImage.create(file)

    # modelni ishga tushurish
    model = load_learner('balls_classifier_model.pkl')

    # prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    #plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)