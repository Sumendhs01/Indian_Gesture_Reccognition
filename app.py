# from sign_translation.helper import *
from sign_translation.translate import translatee
import openai
import streamlit as st
uservid='sign_translation\\utils\\'
import os
api_key = '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'
openai.api_key = api_key







# Streamlit app
st.title("Sentence Generator")
st.title("Better & Easy Communication")
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
# st.video(video_file)
if video_file is not None:
    file_details = {"FileName":video_file.name,"FileType":video_file.type}
    st.write(file_details)
    # vid = load_image(image_file)
    # st.image(img,height=250,width=250)
    vidname=video_file.name
    with open(os.path.join(uservid,video_file.name),"wb") as f: 
      f.write(video_file.getbuffer())         
    st.video(video_file)
    if st.button("Generate Sentence"):
        # Split user input into words
        words =translatee(vidname)

        if words:
            # Construct a prompt
            prompt = f"Form a sentence from the following words in the order of their occurrence: '{', '.join(words)}'."

            # Generate a sentence using GPT-3
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                max_tokens=50,
            )

            # Extract the generated sentence from the response
            generated_sentence = response.choices[0].text

            # Display the generated sentence in a text box with a rectangle shape
            st.text_area("Generated Sentence", generated_sentence, height=150)
            os.remove(vidname)
        else:
            st.warning("Please enter words separated by commas.")
