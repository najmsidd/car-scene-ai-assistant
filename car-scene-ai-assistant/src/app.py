import streamlit as st
from detect import detect_image
from gpt_response import build_prompt, gpt_response
import os
from PIL import Image

st.title("Car scene ai assistant")

uploaded_file = st.file_uploader("Upload an image of the traffic", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image_path = os.path.join("images", uploaded_file.name)

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(image_path, caption="Uploaded traffic image", use_column_width=True)

    if st.button("Analyze image"):

        object_counts, output_file_path= detect_image(image_path)

        st.image(output_file_path, caption="Analyzed Image", use_column_width=True)

        st.subheader("Detected Objects:")
        st.write(object_counts)

        builded_prompt = build_prompt(object_counts)
        st.subheader("Generated Prompt:")
        st.code(builded_prompt)

        gpt_generated_response = gpt_response(builded_prompt)
        st.subheader("GPT Generated response:")
        st.write(gpt_generated_response)

        with open(output_file_path, "rb") as img_file:
            st.download_button(label="Download analyzed image", data=img_file, file_name=output_file_path)





    






