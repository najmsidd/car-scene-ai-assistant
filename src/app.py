import streamlit as st
from detect import detect_image
from gpt_response import build_prompt, gpt_response, build_driver_prompt
import os
from PIL import Image
from video_pipeline import detect_video
from logger import log_analysis
import json

st.title("Car scene ai assistant")

os.makedirs("images", exist_ok=True)
os.makedirs("videos", exist_ok=True)
os.makedirs("detections", exist_ok=True)

option = st.selectbox("Choose input type",("Image", "Video", "History")) 

if option == "Image":
    input_type = "Image"
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
            st.subheader("Scene Description Prompt:")
            st.code(builded_prompt)

            gpt_generated_response = gpt_response(builded_prompt)
            st.subheader("GPT Scene response:")
            st.write(gpt_generated_response)

            driver_prompt = build_driver_prompt(object_counts)
            st.subheader("Driver Recommendations Prompt:")
            st.code(driver_prompt)

            gpt_driver_response = gpt_response(driver_prompt)
            st.subheader("GPT Driver Recommendations")
            st.write(gpt_driver_response)

            log_analysis(
                uploaded_file.name,
                object_counts,
                gpt_generated_response,
                gpt_driver_response,
                input_type
            )

            with open(output_file_path, "rb") as img_file:
                st.download_button(label="Download analyzed image", data=img_file, file_name=output_file_path)

elif option == "Video":
    input_type = "Video"
    uploaded_video = st.file_uploader("Upload a video of the traffic", type=["mp4","avi","mov"])

    if uploaded_video is not None:
        video_path = os.path.join("videos", uploaded_video.name)

        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        st.video(video_path)

        if st.button("Analyze video"):

            output_video_path = os.path.join("detections", f"detected_{uploaded_video.name}")

            object_counts = detect_video(video_path, output_video_path)

            st.success("Video processed successfully!")

            with open(output_video_path, "rb") as vid_file:
                video_bytes = vid_file.read()

            st.video(video_bytes)

            st.subheader("Detected objects in video:")
            st.write(object_counts)

            builded_prompt = build_prompt(object_counts)
            st.subheader("Scene description prompt(video):")
            st.code(builded_prompt)

            gpt_generated_response = gpt_response(builded_prompt)
            st.subheader("GPT Scene response (video)")
            st.write(gpt_generated_response)

            driver_prompt = build_driver_prompt(object_counts)
            st.subheader("GPT Driver recommendations (video")
            st.code(driver_prompt)

            gpt_driver_response = gpt_response(driver_prompt)
            st.subheader("GPT Driver recommendations (video)")
            st.write(gpt_driver_response)

            log_analysis(
                uploaded_video.name,
                object_counts,
                gpt_generated_response,
                gpt_driver_response,
                input_type
            )

            st.download_button(label="Download analyzed video", data=video_bytes, file_name=output_video_path)

elif option == "History":
    st.subheader("Analysis history")

    try:
        with open('history.json', 'r') as f:
            log_data = json.load(f)
    
    except (FileNotFoundError, json.JSONDecodeError):
        st.warning("No analysis log found yet")
        log_data = []

    if not log_data:
        st.info("No records to display")

    else:
        for entry in reversed(log_data):
            st.markdown("---")
            st.markdown(f"**File:** {entry.get('filename', 'N/A')}")
            st.markdown(f"**Type:** {entry.get('input_type', 'Image')}")
            st.markdown(f"**Date:** {entry.get('timestamp', 'Unknown')}")
            
            st.markdown("**Object counts**")
            st.json(entry.get('object_counts',{}))

            st.markdown("**Scene description:**")
            st.write(entry.get('scene_description', "N/A"))

            st.markdown("**Driver recommendations:**")
            st.write(entry.get('driver_recommendations', 'N/A'))   





