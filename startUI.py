import streamlit as st
import base64
from io import BytesIO
from PIL import Image
import requests
import os
from openai import OpenAI

openai_api_key = "aaaa"
client = OpenAI(api_key=openai_api_key)

# Streamlit UI
st.title("블로그 게시글 생성기")

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "home"
if "generated_text" not in st.session_state:
    st.session_state.generated_text = ""

if st.session_state.page == "home":

    st.header("게시글을 쓰기 위한 세팅 화면")

    if "brief_info" not in st.session_state:
        st.session_state.brief_info = ""

    Type = st.radio(
        "원하는 옵션을 선택하세요.",
        ["일상 기록", "제품 소개", "칼럼", "검색"],
        captions=[
            "기록하고 싶은 일상.",
            "편리한 광고 작성.",
            "전문적인 칼럼.",
            "당신만의 고유한 아카이브."
        ]
    )

    if Type == "일상 기록":
        uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:

            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Convert the image to Base64
            buffered = BytesIO()
            image.save(buffered, format="png")  # Adjust format if necessary (e.g., JPEG)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": "여행 중 찍은 사진인데 이를 설명해줘"},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                    },
                    },
                ],
                }
            ],
            max_tokens=500,
            )

            st.session_state.generated_text = response.choices[0].message.content
        
        mood = st.selectbox("Mood", ["활기찬", "우울한", "무던한"])
        date = st.date_input("날짜")
        time = st.time_input("시간대")
        brief_info = st.text_area("간략한 정보", key = "brief_info")
        tone = st.selectbox("말투", ["격식있는", "캐주얼한", "유머러스한", "CUSTOM"])
        generate_btn = st.button("생성")

        if generate_btn:
            st.session_state.page = "result"


    elif Type == "제품 소개":
        uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:

            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Convert the image to Base64
            buffered = BytesIO()
            image.save(buffered, format="png")  # Adjust format if necessary (e.g., JPEG)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": "제품 사진과 관련해서 광고 또는 홍보 글을 작성해줘."},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                    },
                    },
                ],
                }
            ],
            max_tokens=500,
            )

            st.session_state.generated_text = response.choices[0].message.content
    
        brief_info = st.text_area("간략한 정보", key = "brief_info")
        tone = st.selectbox("말투", ["격식있는", "캐주얼한", "유머러스한", "CUSTOM"])
        generate_btn = st.button("생성")

        if generate_btn:
            st.session_state.page = "result"


    elif Type == "칼럼":
        uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:

            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Convert the image to Base64
            buffered = BytesIO()
            image.save(buffered, format="png")  # Adjust format if necessary (e.g., JPEG)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": "brief_info에 입력된 정보를 바탕으로, 사진과 관련된 칼럼을 작성해줘."},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                    },
                    },
                ],
                }
            ],
            max_tokens=500,
            )

            st.session_state.generated_text = response.choices[0].message.content
        
        brief_info = st.text_area("간략한 정보", key = "brief_info")
        tone = st.selectbox("말투", ["격식있는", "캐주얼한", "유머러스한", "CUSTOM"])
        generate_btn = st.button("생성")

        if generate_btn:
            st.session_state.page = "result"



    elif Type == "검색":
        user_search_param = st.number_input("User ID", value=0, step=1, format="%d")
        user_search_btn = st.button("Search")

        if user_search_btn:
            st.write(f"Search submitted for User ID: {user_search_param}")

        keys = list(st.session_state.keys())
        for key in keys:
            if key != 'brief_info':
                st.session_state.pop(key)

elif st.session_state.page == "result":
    st.write(st.session_state.generated_text)
    back_btn = st.button("뒤로 가기")

    if back_btn:
        st.session_state.page = "home"
        st.session_state.generated_text = ""

    