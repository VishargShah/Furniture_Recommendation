# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 23:56:10 2024

@author: P00121384
"""
import os
import streamlit as st
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)

PROJECT_ID = os.environ.get("GCP_PROJECT")  # Your Google Cloud Project ID
LOCATION = os.environ.get("GCP_REGION")  # Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)


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


st.header("Vertex AI Gemini 1.0 API", divider="rainbow")
text_model_pro, multimodal_model_pro = load_models()

tab3, tab4 = st.tabs(
    ["Image Playground", "Video Playground"]
)

with tab3:
    st.write("Using Gemini 1.0 Pro Vision - Multimodal model")
    image_undst, diagrams_undst, recommendations, sim_diff = st.tabs(
        [
            "Furniture recommendation",
            "ER diagrams",
            "Glasses recommendation",
            "Math reasoning",
        ]
    )

    with image_undst:
        st.markdown(
            """In this demo, you will be presented with a scene (e.g., a living room) and will use the Gemini 1.0 Pro Vision model to perform visual understanding. You will see how Gemini 1.0 can be used to recommend an item (e.g., a chair) from a list of furniture options as input. You can use Gemini 1.0 Pro Vision to recommend a chair that would complement the given scene and will be provided with its rationale for such selections from the provided list.
                    """
        )
        
        
        # upload image - Interior Design 
        #with st.form("my-form", clear_on_submit=True):
        #    uploaded_files = st.file_uploader("Choose an Interior image:", type=['png', 'jpg'], accept_multiple_files=False)
        #    submitted = st.form_submit_button("SUBMIT")
        
        # upload image - Catalogue
        #with st.form("my-form", clear_on_submit=True):
        #    uploaded_files = st.file_uploader("Choose an image from Catalogue:", type=['png', 'jpg'], accept_multiple_files=True)
        #    submitted = st.form_submit_button("SUBMIT")
        

        room_image_uri = (
            "gs://github-repo/img/gemini/retail-recommendations/rooms/living_room.jpeg"
        )
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

        room_image_urls = (
            "https://storage.googleapis.com/" + room_image_uri.split("gs://")[1]
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
        chair_1_image = Part.from_uri(chair_1_image_uri, mime_type="image/jpeg")
        chair_2_image = Part.from_uri(chair_2_image_uri, mime_type="image/jpeg")
        chair_3_image = Part.from_uri(chair_3_image_uri, mime_type="image/jpeg")
        chair_4_image = Part.from_uri(chair_4_image_uri, mime_type="image/jpeg")

        st.image(room_image_urls, width=350, caption="Image of a living room")
        st.image(
            [
                chair_1_image_urls,
                chair_2_image_urls,
                chair_3_image_urls,
                chair_4_image_urls,
            ],
            width=200,
            caption=["Chair 1", "Chair 2", "Chair 3", "Chair 4"],
        )

        st.write(
            "Our expectation: Recommend a chair that would complement the given image of a living room."
        )
        content = [
            "Consider the following chairs:",
            "chair 1:",
            chair_1_image,
            "chair 2:",
            chair_2_image,
            "chair 3:",
            chair_3_image,
            "and",
            "chair 4:",
            chair_4_image,
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
        er_diag_uri = (
            "gs://github-repo/img/gemini/retail-recommendations/rooms/living_room.jpeg"
        )
        er_diag_url = (
            "https://storage.googleapis.com/" + room_image_uri.split("gs://")[1]
        )


        st.write(
            "Gemini 1.0 Pro Vision multimodal capabilities empower it to comprehend diagrams and describe it. The following example demonstrates how Gemini 1.0 can decipher an Interior Design"
        )
        er_diag_img = Part.from_uri(er_diag_uri, mime_type="image/jpeg")
        st.image(er_diag_url, width=350, caption="Image of a Interior Design")
        st.write(
            "Our expectation: Describe the Interior Design Image."
        )
        prompt = """Document the objects present in Interior design and describe the entire interior room.
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

    with recommendations:
        compare_img_1_uri = (
            "gs://github-repo/img/gemini/multimodality_usecases_overview/glasses1.jpg"
        )
        compare_img_2_uri = (
            "gs://github-repo/img/gemini/multimodality_usecases_overview/glasses2.jpg"
        )

        compare_img_1_url = (
            "https://storage.googleapis.com/" + compare_img_1_uri.split("gs://")[1]
        )
        compare_img_2_url = (
            "https://storage.googleapis.com/" + compare_img_2_uri.split("gs://")[1]
        )

        st.write(
            """Gemini 1.0 Pro Vision is capable of image comparison and providing recommendations. This may be useful in industries like e-commerce and retail.
                    Below is an example of choosing which pair of glasses would be better suited to various face types:"""
        )
        compare_img_1_img = Part.from_uri(compare_img_1_uri, mime_type="image/jpeg")
        compare_img_2_img = Part.from_uri(compare_img_2_uri, mime_type="image/jpeg")
        face_type = st.radio(
            "What is your face shape?",
            ["Oval", "Round", "Square", "Heart", "Diamond"],
            key="face_type",
            horizontal=True,
        )
        output_type = st.radio(
            "Select the output type",
            ["text", "table", "json"],
            key="output_type",
            horizontal=True,
        )
        st.image(
            [compare_img_1_url, compare_img_2_url],
            width=350,
            caption=["Glasses type 1", "Glasses type 2"],
        )
        st.write(
            f"Our expectation: Suggest which glasses type is better for the {face_type} face shape"
        )
        content = [
            f"""Which of these glasses you recommend for me based on the shape of my face:{face_type}?
           I have an {face_type} shape face.
           Glasses 1: """,
            compare_img_1_img,
            """
           Glasses 2: """,
            compare_img_2_img,
            f"""
           Explain how you reach out to this decision.
           Provide your recommendation based on my face shape, and reasoning for each in {output_type} format.
           """,
        ]
        tab1, tab2 = st.tabs(["Response", "Prompt"])
        compare_img_description = st.button(
            "Generate recommendation!", key="compare_img_description"
        )
        with tab1:
            if compare_img_description and content:
                with st.spinner(
                    "Generating recommendations using Gemini 1.0 Pro Vision..."
                ):
                    response = get_gemini_pro_vision_response(
                        multimodal_model_pro, content
                    )
                    st.markdown(response)
        with tab2:
            st.write("Prompt used:")
            st.text(content)

    with sim_diff:
        math_image_uri = "gs://github-repo/img/gemini/multimodality_usecases_overview/math_beauty.jpg"
        math_image_url = (
            "https://storage.googleapis.com/" + math_image_uri.split("gs://")[1]
        )
        st.write(
            "Gemini 1.0 Pro Vision can also recognize math formulas and equations and extract specific information from them. This capability is particularly useful for generating explanations for math problems, as shown below."
        )
        math_image_img = Part.from_uri(math_image_uri, mime_type="image/jpeg")
        st.image(math_image_url, width=350, caption="Image of a math equation")
        st.markdown(
            """
                Our expectation: Ask questions about the math equation as follows:
                - Extract the formula.
                - What is the symbol right before Pi? What does it mean?
                - Is this a famous formula? Does it have a name?
                    """
        )
        prompt = """
Follow the instructions.
Surround math expressions with $.
Use a table with a row for each instruction and its result.

INSTRUCTIONS:
- Extract the formula.
- What is the symbol right before Pi? What does it mean?
- Is this a famous formula? Does it have a name?
"""
        tab1, tab2 = st.tabs(["Response", "Prompt"])
        math_image_description = st.button(
            "Generate answers!", key="math_image_description"
        )
        with tab1:
            if math_image_description and prompt:
                with st.spinner(
                    "Generating answers for formula using Gemini 1.0 Pro Vision..."
                ):
                    response = get_gemini_pro_vision_response(
                        multimodal_model_pro, [math_image_img, prompt]
                    )
                    st.markdown(response)
                    st.markdown("\n\n\n")
        with tab2:
            st.write("Prompt used:")
            st.text(prompt)

with tab4:
    st.write("Using Gemini 1.0 Pro Vision - Multimodal model")

    vide_desc, video_tags, video_highlights, video_geolocation = st.tabs(
        ["Video description", "Video tags", "Video highlights", "Video geolocation"]
    )

    with vide_desc:
        st.markdown(
            """Gemini 1.0 Pro Vision can also provide the description of what is going on in the video:"""
        )
        vide_desc_uri = "gs://github-repo/img/gemini/multimodality_usecases_overview/mediterraneansea.mp4"
        video_desc_url = (
            "https://storage.googleapis.com/" + vide_desc_uri.split("gs://")[1]
        )
        if vide_desc_uri:
            vide_desc_img = Part.from_uri(vide_desc_uri, mime_type="video/mp4")
            st.video(video_desc_url)
            st.write("Our expectation: Generate the description of the video")
            prompt = """Describe what is happening in the video and answer the following questions: \n
            - What am I looking at? \n
            - Where should I go to see it? \n
            - What are other top 5 places in the world that look like this?
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
        video_tags_uri = "gs://github-repo/img/gemini/multimodality_usecases_overview/photography.mp4"
        video_tags_url = (
            "https://storage.googleapis.com/" + video_tags_uri.split("gs://")[1]
        )
        if video_tags_url:
            video_tags_img = Part.from_uri(video_tags_uri, mime_type="video/mp4")
            st.video(video_tags_url)
            st.write("Our expectation: Generate the tags for the video")
            prompt = """Answer the following questions using the video only:
                        1. What is in the video?
                        2. What objects are in the video?
                        3. What is the action in the video?
                        4. Provide 5 best tags for this video?
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
    with video_highlights:
        st.markdown(
            """Below is another example of using Gemini 1.0 Pro Vision to ask questions about objects, people or the context, as shown in the video about Pixel 8 below:"""
        )
        video_highlights_uri = (
            "gs://github-repo/img/gemini/multimodality_usecases_overview/pixel8.mp4"
        )
        video_highlights_url = (
            "https://storage.googleapis.com/" + video_highlights_uri.split("gs://")[1]
        )
        if video_highlights_url:
            video_highlights_img = Part.from_uri(
                video_highlights_uri, mime_type="video/mp4"
            )
            st.video(video_highlights_url)
            st.write("Our expectation: Generate the highlights for the video")
            prompt = """Answer the following questions using the video only:
What is the profession of the girl in this video?
Which all features of the phone are highlighted here?
Summarize the video in one paragraph.
Provide the answer in table format.
            """
            tab1, tab2 = st.tabs(["Response", "Prompt"])
            video_highlights_description = st.button(
                "Generate video highlights", key="video_highlights_description"
            )
            with tab1:
                if video_highlights_description and prompt:
                    with st.spinner(
                        "Generating video highlights using Gemini 1.0 Pro Vision ..."
                    ):
                        response = get_gemini_pro_vision_response(
                            multimodal_model_pro, [prompt, video_highlights_img]
                        )
                        st.markdown(response)
                        st.markdown("\n\n\n")
            with tab2:
                st.write("Prompt used:")
                st.write(prompt, "\n", "{video_data}")

    with video_geolocation:
        st.markdown(
            """Even in short, detail-packed videos, Gemini 1.0 Pro Vision can identify the locations."""
        )
        video_geolocation_uri = (
            "gs://github-repo/img/gemini/multimodality_usecases_overview/bus.mp4"
        )
        video_geolocation_url = (
            "https://storage.googleapis.com/" + video_geolocation_uri.split("gs://")[1]
        )
        if video_geolocation_url:
            video_geolocation_img = Part.from_uri(
                video_geolocation_uri, mime_type="video/mp4"
            )
            st.video(video_geolocation_url)
            st.markdown(
                """Our expectation: \n
            Answer the following questions from the video:
                - What is this video about?
                - How do you know which city it is?
                - What street is this?
                - What is the nearest intersection?
            """
            )
            prompt = """Answer the following questions using the video only:
            What is this video about?
            How do you know which city it is?
            What street is this?
            What is the nearest intersection?
            Answer the following questions in a table format with question and answer as columns.
            """
            tab1, tab2 = st.tabs(["Response", "Prompt"])
            video_geolocation_description = st.button(
                "Generate", key="video_geolocation_description"
            )
            with tab1:
                if video_geolocation_description and prompt:
                    with st.spinner(
                        "Generating location tags using Gemini 1.0 Pro Vision ..."
                    ):
                        response = get_gemini_pro_vision_response(
                            multimodal_model_pro, [prompt, video_geolocation_img]
                        )
                        st.markdown(response)
                        st.markdown("\n\n\n")
            with tab2:
                st.write("Prompt used:")
                st.write(prompt, "\n", "{video_data}")