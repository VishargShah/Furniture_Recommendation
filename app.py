# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 23:56:10 2024

@author: P00121384
"""
import os
import PIL.Image
from pathlib import Path as p
import streamlit as st
#from generate_file import image_creation
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
import google.auth
from google.cloud import storage

PROJECT_ID = os.environ.get("GCP_PROJECT")  # Your Google Cloud Project ID
LOCATION = os.environ.get("GCP_REGION")  # Your Google Cloud Project Region
BUCKET_URI = "gs://image_video_storage"
vertexai.init(project=PROJECT_ID, location=LOCATION)

#Initializing Directory
data_folder = p.cwd() / "video_files"
p(data_folder).mkdir(parents=True, exist_ok=True)
data_folder = p.cwd() / "image_description"
p(data_folder).mkdir(parents=True, exist_ok=True)
data_folder = p.cwd() / "image_recommendation"
p(data_folder).mkdir(parents=True, exist_ok=True)

@st.cache_resource
def load_models():
    """
    Load the generative models for text and multimodal generation.

    Returns:
        Tuple: A tuple containing the text model and multimodal model.
    """
    text_model_pro = GenerativeModel("gemini-1.0-pro")
    multimodal_model_pro = GenerativeModel("gemini-1.0-pro-vision")
    return text_model_pro, multimodal_model_pro


def get_gemini_pro_text_response(
    model: GenerativeModel,
    contents: str,
    generation_config: GenerationConfig,
    stream: bool = True,
):
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    responses = model.generate_content(
        prompt,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=stream,
    )

    final_response = []
    for response in responses:
        try:
            # st.write(response.text)
            final_response.append(response.text)
        except IndexError:
            # st.write(response)
            final_response.append("")
            continue
    return " ".join(final_response)


def get_gemini_pro_vision_response(
    model, prompt_list, generation_config={}, stream: bool = True
):
    generation_config = {"temperature": 0.1, "max_output_tokens": 2048}
    responses = model.generate_content(
        prompt_list, generation_config=generation_config, stream=stream
    )
    final_response = []
    for response in responses:
        try:
            final_response.append(response.text)
        except IndexError:
            pass
    return "".join(final_response)


########################## Page configuration #############################
fp = open("./streamlit/AP.jpg","rb")
image = PIL.Image.open(fp)
st.set_page_config(
    page_title="Beyond BHS",
    page_icon=image,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "## Get the AI based recommendation of the AP SKU and descriptions of interior\n **Contact for updates** : Team Asian Paints -6A"
    })

st.header("Beyond BHS: Recommendation & Description🪄", divider="rainbow")
text_model_pro, multimodal_model_pro = load_models()

tab3, tab4 = st.tabs(
    ["Image Playground", "Video Playground"]
)

with tab3:
    st.write("Using Gemini 1.0 Pro Vision - Multimodal model")
    image_undst, diagrams_undst = st.tabs(
        [
            "Furniture recommendation",
            "Interior Design Description",
        ]
    )

    with image_undst:
        st.markdown(
            """In this demo, you will be presented with a scene (e.g., a living room) and will use the Gemini 1.0 Pro Vision model to perform visual understanding. You will see how Gemini 1.0 can be used to recommend an item (e.g., a chair) from a list of furniture options as input. You can use Gemini 1.0 Pro Vision to recommend a chair that would complement the given scene and will be provided with its rationale for such selections from the provided list.
                    """
        )
        
        # Audio Uploading Tab
        bucket_name = 'image_video_storage'
        # 1. Authenticate to Google Cloud
        credentials, project = google.auth.default()
        # 2. Create a storage client
        storage_client = storage.Client(project=PROJECT_ID)
        # 3. Get a reference to the bucket (check existence)
        bucket = storage_client.bucket(bucket_name)
        
        
        room_image_uri = "gs://image_video_storage/recommendation_on.jpeg"
        chair_1_image_uri = (
            "gs://github-repo/img/gemini/retail-recommendations/furnitures/chair1.jpeg"
        )
        chair_2_image_uri = (
            "gs://github-repo/img/gemini/retail-recommendations/furnitures/chair2.jpeg"
        )
        chair_3_image_uri = (
            "gs://github-repo/img/gemini/retail-recommendations/furnitures/chair3.jpeg"
        )
        chair_4_image_uri = (
            "gs://github-repo/img/gemini/retail-recommendations/furnitures/chair4.jpeg"
        )

        chair_1_image_urls = (
            "https://storage.googleapis.com/" + chair_1_image_uri.split("gs://")[1]
        )
        chair_2_image_urls = (
            "https://storage.googleapis.com/" + chair_2_image_uri.split("gs://")[1]
        )
        chair_3_image_urls = (
            "https://storage.googleapis.com/" + chair_3_image_uri.split("gs://")[1]
        )
        chair_4_image_urls = (
            "https://storage.googleapis.com/" + chair_4_image_uri.split("gs://")[1]
        )

        room_image = Part.from_uri(room_image_uri, mime_type="image/jpeg")
        SKU_1_image = Part.from_uri(chair_1_image_uri, mime_type="image/jpeg")
        SKU_2_image = Part.from_uri(chair_2_image_uri, mime_type="image/jpeg")
        SKU_3_image = Part.from_uri(chair_3_image_uri, mime_type="image/jpeg")
        SKU_4_image = Part.from_uri(chair_4_image_uri, mime_type="image/jpeg")

        
        # Get the blob (file) object
        room_image_uri_regex = room_image_uri.replace("gs://image_video_storage/","")
        blob = bucket.blob(room_image_uri_regex)
        blob.download_to_filename('image_description/recommendation_on.jpeg')
        st.image('image_recommendation/recommendation_on.jpeg', width=500, caption="Image of a living room")
        
        st.image(
            [
                chair_1_image_urls,
                chair_2_image_urls,
                chair_3_image_urls,
                chair_4_image_urls,
            ],
            width=200,
            caption=["SKU 1", "SKU 2", "SKU 3", "SKU 4"],
        )

        st.write(
            "Our expectation: Recommend a chair that would complement the given image of a living room."
        )
        content = [
            "Consider the following SKU:",
            "SKU 1:",
            SKU_1_image,
            "SKU 2:",
            SKU_2_image,
            "SKU 3:",
            SKU_3_image,
            "and",
            "SKU 4:",
            SKU_4_image,
            "\n"
            "For each chair, explain why it would be suitable or not suitable for the following room:",
            room_image,
            "Only recommend for the room provided and not other rooms. Provide your recommendation in a table format with chair name and reason as columns.",
        ]

        tab1, tab2 = st.tabs(["Response", "Prompt"])
        generate_image_description = st.button(
            "Generate recommendation....", key="generate_image_description"
        )
        with tab1:
            if generate_image_description and content:
                with st.spinner(
                    "Generating recommendation using Gemini 1.0 Pro Vision ..."
                ):
                    response = get_gemini_pro_vision_response(
                        multimodal_model_pro, content
                    )
                    st.markdown(response)
        with tab2:
            st.write("Prompt used:")
            st.text(content)

    with diagrams_undst:
        er_diag_uri = "gs://image_video_storage/describe_image.jpeg"
        st.write(
            "Gemini 1.0 Pro Vision multimodal capabilities empower it to comprehend diagrams and describe it. The following example demonstrates how Gemini 1.0 can decipher an Interior Design"
        )
        er_diag_img = Part.from_uri(er_diag_uri, mime_type="image/jpeg")
        # Audio Uploading Tab
        bucket_name = 'image_video_storage'
        # 1. Authenticate to Google Cloud
        credentials, project = google.auth.default()
        # 2. Create a storage client
        storage_client = storage.Client(project=PROJECT_ID)
        # 3. Get a reference to the bucket (check existence)
        bucket = storage_client.bucket(bucket_name)
        # Get the blob (file) object
        er_diag_uri_regex = er_diag_uri.replace("gs://image_video_storage/","")
        blob = bucket.blob(er_diag_uri_regex)
        blob.download_to_filename('image_description/describe_image.jpeg')
        st.image('image_description/describe_image.jpeg', width=700, caption="Image of a Interior Design")
        st.write(
            "Our expectation: Describe the Interior Design Image."
        )
        prompt = """
        Consider yourself as an experience in Interior Design space. You are given the image of interior room and you need to describe in detail. Document the objects present in Interior design and describe the entire interior room. Be specific in each and evry aspect of the design.
        """
        tab1, tab2 = st.tabs(["Response", "Prompt"])
        er_diag_img_description = st.button("Generate!", key="er_diag_img_description")
        with tab1:
            if er_diag_img_description and prompt:
                with st.spinner("Generating..."):
                    response = get_gemini_pro_vision_response(
                        multimodal_model_pro, [er_diag_img, prompt]
                    )
                    st.markdown(response)
        with tab2:
            st.write("Prompt used:")
            st.text(prompt + "\n" + "input_image")

with tab4:
    st.write("Using Gemini 1.0 Pro Vision - Multimodal model")
    vide_desc, video_tags = st.tabs(
        ["Video description", "Video tags"]
    )
    with vide_desc:
        st.markdown(
            """Gemini 1.0 Pro Vision can also provide the description of interior design in the video:"""
        )
        vide_desc_uri = "gs://image_video_storage/video.mp4"
        # Audio Uploading Tab
        bucket_name = 'image_video_storage'
        # 1. Authenticate to Google Cloud
        credentials, project = google.auth.default()
        # 2. Create a storage client
        storage_client = storage.Client(project=PROJECT_ID)
        # 3. Get a reference to the bucket (check existence)
        bucket = storage_client.bucket(bucket_name)
        # Get the blob (file) object
        vide_desc_uri_regex = vide_desc_uri.replace("gs://image_video_storage/","")
        blob = bucket.blob(vide_desc_uri_regex)
        blob.download_to_filename('video_files/video.mp4')
        if vide_desc_uri:
            vide_desc_img = Part.from_uri(vide_desc_uri, mime_type="video/mp4")
            st.video('video_files/video.mp4', format="video/mp4")
            st.write("Our expectation: Generate the description of the video")
            prompt = """
            Describe the interior design of the space in details for each of the following mentioned points: \n
            - What is the kind of design & theme in detail? \n
            - Where are the furniture and decor items present in detail? \n
            - What is the colour theme, ligting, rythm in detail?

            """
            tab1, tab2 = st.tabs(["Response", "Prompt"])
            vide_desc_description = st.button(
                "Generate video description", key="vide_desc_description"
            )
            with tab1:
                if vide_desc_description and prompt:
                    with st.spinner(
                        "Generating video description using Gemini 1.0 Pro Vision ..."
                    ):
                        response = get_gemini_pro_vision_response(
                            multimodal_model_pro, [prompt, vide_desc_img]
                        )
                        st.markdown(response)
                        st.markdown("\n\n\n")
            with tab2:
                st.write("Prompt used:")
                st.write(prompt, "\n", "{video_data}")

    with video_tags:
        st.markdown(
            """Gemini 1.0 Pro Vision can also extract tags throughout a video, as shown below:."""
        )
        video_tags_uri = "gs://image_video_storage/video.mp4"
        if video_tags_uri:
            video_tags_img = Part.from_uri(video_tags_uri, mime_type="video/mp4")
            st.video('video_files/video.mp4', format="video/mp4")
            st.write("Our expectation: Generate the tags for the video")
            prompt = """Answer the following questions using the video only:
                        1. What is in the video?
                        2. What are the furniture objects present in the video?
                        3. What are the light fittings presnt in the video?
                        4. What are the decorative object present in the video?
                        5. What is the colour of the walls?
                        Give the answer in the table format with question and answer as columns.
            """
            tab1, tab2 = st.tabs(["Response", "Prompt"])
            video_tags_description = st.button(
                "Generate video tags", key="video_tags_description"
            )
            with tab1:
                if video_tags_description and prompt:
                    with st.spinner(
                        "Generating video description using Gemini 1.0 Pro Vision ..."
                    ):
                        response = get_gemini_pro_vision_response(
                            multimodal_model_pro, [prompt, video_tags_img]
                        )
                        st.markdown(response)
                        st.markdown("\n\n\n")
            with tab2:
                st.write("Prompt used:")
                st.write(prompt, "\n", "{video_data}")