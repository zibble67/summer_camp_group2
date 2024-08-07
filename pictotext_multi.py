import streamlit as st
import base64
from io import BytesIO
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS, GPSTAGS
from openai import OpenAI
import time
import pyperclip
from datetime import datetime, timedelta
from geopy.distance import geodesic

openai_api_key = ""
client = OpenAI(api_key=openai_api_key)

# Streamlit UI
st.title("블로그 게시글 생성기")

################### 변수 초기화 & 상수 정의 ########################
# 1분 이내와 근접한 위치를 정의하는 상수
TIME_THRESHOLD = timedelta(minutes=30)
DISTANCE_THRESHOLD = 100  # 거리 임계값 (단위: km, 여기서는 100m)

if "page" not in st.session_state:
    st.session_state.page = "home"
if "blog_content" not in st.session_state:
    st.session_state.blog_content = []
if "group_info" not in st.session_state:
    st.session_state.group_info = []
if "image" not in st.session_state:
    st.session_state.image = None
if "Type" not in st.session_state:
    st.session_state.Type = ""
if 'generated' not in st.session_state:
    st.session_state['generated'] = False
if 'info' not in st.session_state:
    st.session_state.info = []


################## 필요한 함수 정의 #####################
def resize_image(image, max_size_kb):   #해상도 조절 함수
    # 이미지의 현재 크기 계산
    img_bytes = BytesIO()
    image.save(img_bytes, format='JPEG')
    img_size_kb = len(img_bytes.getvalue()) / 1024
    
    # 이미지 크기가 max_size_kb 이상이면 해상도 조정
    while img_size_kb > max_size_kb:
        # 이미지 해상도 줄이기
        image = image.resize((int(image.size[0] * 0.9), int(image.size[1] * 0.9)))
        img_bytes = BytesIO()
        image.save(img_bytes, format='JPEG')
        img_size_kb = len(img_bytes.getvalue()) / 1024
    
    return img_bytes


def generate_blog(Type, photo, info):      # 블로그 글 작성 함수
    filtered_dict = {k: v for k, v in photo.items() if k != 'img_base64'}

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": Type},
            {"role": "user", "content":[
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{photo['img_base64']}",
                    },
                    },
                {"type": "text", "text": f"Please write a blog writing in korean according to images and the following information: {filtered_dict}, {info}."}, 
            ]
            }
        ],
        max_tokens=500,
        stream=False
    )
    return response.choices[0].message.content

def refine_blog(blog_content, additional_prompt):  # 블로그 글 수정 함수
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Modify the blog according to additional_prompt"},
            {"role": "assistant", "content": blog_content},
            {"role": "user", "content": additional_prompt}
        ],
        max_tokens=500,
        stream=False
    )

    return response.choices[0].message.content

def get_exif_data(image):   
    exif_data = {}
    try:
        exif = image._getexif()
        if exif is not None:
            for tag, value in exif.items():
                tag_name = TAGS.get(tag, tag)
                exif_data[tag_name] = value
    except AttributeError:
        st.error("이미지에서 EXIF 데이터를 추출할 수 없습니다.")
    return exif_data

def correct_image_orientation(image, exif_data):    # 이미지 회전 보정 함수
    if 'Orientation' in exif_data:
        orientation = exif_data['Orientation']
        if orientation == 3:
            image = image.rotate(180, expand=True)
        elif orientation == 6:
            image = image.rotate(270, expand=True)
        elif orientation == 8:
            image = image.rotate(90, expand=True)
    return image

def extract_datetime(exif_data):    # 이미지 시간 데이터 추출 함수
    if 'DateTimeOriginal' in exif_data:
        date_time_str = exif_data['DateTimeOriginal']
        try:
            return datetime.strptime(date_time_str, "%Y:%m:%d %H:%M:%S")
        except ValueError:
            return None
    return None

def get_gps_info(exif_data):    # 사진 속 장소 gps 데이터 얻는 함수
    gps_info = {}
    if 'GPSInfo' in exif_data:
        for key in exif_data['GPSInfo'].keys():
            decode = GPSTAGS.get(key, key)
            gps_info[decode] = exif_data['GPSInfo'][key]
    return gps_info

def get_coordinates(gps_info):  # 사진 속 장소 위도/경도 데이터 얻는 함수
    def convert_to_degrees(value):
        d, m, s = value[0], value[1], value[2]
        return d + (m / 60.0) + (s / 3600.0)

    lat = None
    lon = None
    if 'GPSLatitude' in gps_info and 'GPSLatitudeRef' in gps_info:
        lat = convert_to_degrees(gps_info['GPSLatitude'])
        if gps_info['GPSLatitudeRef'] != 'N':
            lat = -lat

    if 'GPSLongitude' in gps_info and 'GPSLongitudeRef' in gps_info:
        lon = convert_to_degrees(gps_info['GPSLongitude'])
        if gps_info['GPSLongitudeRef'] != 'E':
            lon = -lon

    return lat, lon

def group_photos_by_time_and_location(photos_info):     # 시간과 장소로 사진 그룹화 함수
    groups = []
    for photo in photos_info:
        matched = False
        for group in groups:
            time_diff = abs(group[-1]['datetime'] - photo['datetime'])
            if time_diff <= TIME_THRESHOLD:
                if photo['lat'] is not None and photo['lon'] is not None:
                    last_photo = group[-1]
                    distance = geodesic((last_photo['lat'], last_photo['lon']), (photo['lat'], photo['lon'])).km
                    if distance <= DISTANCE_THRESHOLD:
                        group.append(photo)
                        matched = True
                        break
        if not matched:
            groups.append([photo])
    return groups


######################### 시작 UI ###########################
if st.session_state.page == "home":

    Type = st.radio(
        "원하는 옵션을 선택하세요.",
        ["일상 기록", "제품 소개", "칼럼", "검색"],
        captions=[
            "기록하고 싶은 일상.",
            "편리한 광고 작성.",
            "전문적인 칼럼.",
            "당신만의 고유한 아카이브.(미구현)"
        ], 
    )
    st.session_state.Type = Type

    if Type in ["일상 기록", "제품 소개", "칼럼"]:
        uploaded_files = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files is not None:
            photos_info = []

            for uploaded_file in uploaded_files:

                # Display the uploaded image
                image = Image.open(uploaded_file)
                exif_data = get_exif_data(image)

                # 이미지 회전 보정
                image = correct_image_orientation(image, exif_data)
                
                date_time = extract_datetime(exif_data)
                gps_info = get_gps_info(exif_data)
                lat, lon = get_coordinates(gps_info) if gps_info else (None, None)

                # Convert the image to Base64
                buffered = BytesIO()
                image.save(buffered, format="png")  # Adjust format if necessary (e.g., JPEG)
                img_base64 = base64.b64encode(buffered.getvalue()).decode()

                photos_info.append({
                    "filename": uploaded_file.name,
                    "datetime": date_time,
                    "lat": lat,
                    "lon": lon,
                    "image": image,
                    "img_base64": img_base64
                })
            
            # 촬영 시간순으로 정렬
            photos_info.sort(key=lambda x: x['datetime'] or datetime.min)
            
            # 근접한 시간과 위치에 따라 사진 그룹화
            grouped_photos = group_photos_by_time_and_location(photos_info)

            # 그룹화된 사진 출력 (같은 행에 배치)
            for i, group in enumerate(grouped_photos):
                st.subheader(f"그룹 {i+1}")
                cols = st.columns(len(group))  # 그룹 내 사진 수에 따라 컬럼 생성

                for idx, info in enumerate(group):
                    with cols[idx]:
                        st.image(info["image"], caption=f"{info['filename']} (촬영 시간: {info['datetime']})", use_column_width=True)
                        if not info["lat"] or not info["lon"]:
                            st.write("위치 정보를 찾을 수 없습니다.")
                        text = st.text_area(f"{info['filename']}에 대한 메모를 입력하세요.", key=f"memo_{i}_{idx}")
                        info['text'] = text
                st.session_state.group_info.append(group)
                st.write("---")  # 구분선 추가

        if Type == "일상 기록":
            mood = st.selectbox("Mood", ["활기찬", "우울한", "무던한"])
            st.session_state.info.append(mood)

        tone = st.selectbox("말투", ["격식있는", "캐주얼한", "유머러스한", "CUSTOM"])
        st.session_state.info.append(tone)

        if st.button("생성"):
            st.session_state.page = "result"
            st.rerun()  # Immediately rerun the script

    elif Type == "검색":
        user_search_param = st.number_input("User ID", value=0, step=1, format="%d")

        if st.button("Search"):
            st.write(f"Search submitted for User ID: {user_search_param}")

        keys = list(st.session_state.keys())
        for key in keys:
            if key != 'brief_info':
                st.session_state.pop(key)


#################### 중간 UI #######################
elif st.session_state.page == "result":

    if st.session_state.Type == "일상 기록":
        txt = "You are a blog writer who record travel or daily life."
    elif st.session_state.Type == "제품 소개":
        txt = "You are a blog writer who writes an introduction or ad for a product."
    elif st.session_state.Type == "칼럼":
        txt = "You are a blog writer who specializes in writing brief_info."

    if st.session_state.group_info is not None:
        with st.spinner('블로그 글 생성중...'):
            for group in st.session_state.group_info:
                for photo in group:
                    blog_content = generate_blog(txt, photo, st.session_state.info)
                    st.session_state.blog_content.append(blog_content)
            st.session_state['generated'] = True

    # 생성 완료 시
    if st.session_state.get('generated'):
        for content in st.session_state.blog_content:
            st.write(content)

        # 클립보드로 복사 기능 추가
        if st.button("복사"):
            pyperclip.copy(st.session_state.blog_content)
            print(st.session_state.blog_content)
            st.write("클립보드로 복사되었습니다.")

    # 추가 수정 프롬프팅
    additional_prompt = st.text_area("수정을 원한다면 추가 정보를 입력하세요.")

    # 버튼 위치 조정
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])  # Adjust column ratios as needed

    with col1:
        if st.button("수정하기"):
            st.rerun()  # Immediately rerun the script
            with st.spinner('블로그 글 수정중...'):
                refined_content = refine_blog(st.session_state.blog_content, additional_prompt)
                st.session_state['blog_content'] = refined_content
    with col4:
        if st.button("뒤로 가기"):
            st.session_state.page = "home"
            st.session_state.generated_text = ""
            st.rerun()  # Immediately rerun the script
