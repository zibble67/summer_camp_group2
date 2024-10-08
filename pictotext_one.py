import streamlit as st
import base64
from io import BytesIO
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from openai import OpenAI
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
    st.session_state.blog_content = ""
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "image" not in st.session_state:
    st.session_state.image = None
if "Type" not in st.session_state:
    st.session_state.Type = ""
if 'generated' not in st.session_state:
    st.session_state['generated'] = False
if 'info' not in st.session_state:
    st.session_state.info = {}
if 'refine' not in st.session_state:
    st.session_state['refine'] = False


################## 필요한 함수 정의 #####################
def generate_blog(prompt):      # 블로그 글 작성 함수
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": [
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                    },
                    },
                    {"type": "text", "text": f"{prompt}"},
                ],
            }
        ],
        #max_tokens=1000,
        stream=False
    )

    return response.choices[0].message.content

def refine_blog(blog_content, additional_prompt):  # 블로그 글 수정 함수
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Modify the blog according to additional_prompt"},
            {"role": "assistant", "content": blog_content},
            {"role": "user", "content": f"Keep the contents of the existing text(blog_content) as much as possible, but modify it appropriately by reflecting {additional_prompt}, the information given by the user."}
        ],
        #max_tokens=1000,
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
        uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
        st.session_state.uploaded_file = uploaded_file
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)

        if Type == "일상 기록":
            mood = st.selectbox("Mood", ["활기찬", "우울한", "무던한"])
            st.session_state.info['mood'] = mood

        brief_info = st.text_area("간략한 정보")
        st.session_state.info['brief_info'] = brief_info
        tone = st.selectbox("말투", ["격식있는", "캐주얼한", "유머러스한", "CUSTOM"])
        st.session_state.info['tone'] = tone

        if uploaded_file is not None:
            photos_info = []

            exif_data = get_exif_data(image)

            # 이미지 회전 보정
            image = correct_image_orientation(image, exif_data)
            st.session_state.image = image
            
            date_time = extract_datetime(exif_data)
            gps_info = get_gps_info(exif_data)
            lat, lon = get_coordinates(gps_info) if gps_info else (None, None)

            photos_info.append({
                "filename": uploaded_file.name,
                "datetime": date_time,
                "lat": lat,
                "lon": lon,
                "image": image
            })
            st.session_state.info['date_time'] = date_time

        if st.button("생성"):
            st.session_state.page = "result"
            st.session_state['refine'] = False
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
        prompt = f'''너는 인기있는 여행블로거의 역할을 맡을꺼야. 블로그 글을 써야하는데, 아래 조건과 추가 명령 사항을 참고해서, 입력된 사진에 알맞는 글을 작성해줘. 
        그 글은, 여행 다녀온 후기에 맞는 글이어야 해. 지금이 5월이라고 생각하고 화이팅해서 작성해줘 고마워. 
        가장 중요한 것은, 예시를 참고하여 형식적인 GPT 글이 아닌, 사람의 속마음이 담겨져서 사람이 작성한 글처럼 보여야해.
        아, 그리고 이모지(emoji)를 중간중간에 3개 가량 꼭 넣어줘. 다른 설명은 할 필요 없고, 블로그 글만 써줘!
        글 작성을 완료하기전에, 예시1 과 예시2와 형식이 비슷한지 꼭 확인해줘. 저 예시들과 비슷하게 나왔으면 좋겠어

        # 조건
        - 당시 기분은 {st.session_state.info["mood"]}
        - 여행 다녀온 날짜는 {st.session_state.info["date_time"]}
        - 여행 블로그 글의 어투는 {st.session_state.info["tone"]}으로
        - (optional) {st.session_state.info["brief_info"]}이 사용자가 준 힌트 (비었다면 무시해도 됨.)
        - 반말을 기본으로
        - 글자수는 사진 1장당 약 300자 에서 500자 사이
        - 중요한 정보는 글자색 다르게 혹은 굵기를 진하게 하여 표시
        - 구체적인 수치를 기본으로
        - 한문단에 2줄 넘게 작성되면 안됨
        
        # 추가명령
        - 사진이 어떤 사진인지 파악하여 그에 알맞는 글을 생성해낸다.
        (만약 음식 사진이라면, {st.session_state.info["mood"]}를 참고하여 음식의 맛에 대한 평가가 담긴 글이어야하고, 사용자가 간략한 정보를 줬다면, {st.session_state.info["brief_info"]}를 참고하여)
        - 마지막에 p.s. 하며 간단한 한마디를 추가하기

        # 예시1
        확실히 일본여행은 드럭스토어 털어오는 재미를 빼놓고는 얘기가 안되죠..! 이번에 접이식 폴딩백 하나 챙겨간거 가득 쇼핑템으로 담아왔는데요.
        특히 간단한 간식류나 먹거리가 괜찮은게 많아서 출발할때를 대비했을때 짐이 엄청 불었어요 ㅋㅋ
        2박 3일이라고 우습게 볼게 아니라
        캐리어 큰거 챙겨간게 그나마 도움되었네요 :)

        # 예시2
        독일 베를린에 위치한 BRLO 브루어리는 베를린 중심가에서 약 15분 거리에 있는 글라이스드라이에크 공원 바로 옆인 크로이츠베르크 지역에 위치해 있어요.
        글라이스드라이에크 공원은 현지인들에게도 인기 있는 산책로로 날씨 좋은 날에는 공원 산책하고 Beer 한잔하면서 힐링하기 딱 좋답니다.
        근처에 지하철역도 있어서 대중교통 타고 쉽게 이동할 수 있어요.
        주변에 관광지나 맛집들도 많아서 볼거리가 많은 곳이니 기억해두세요.

        '''
    elif st.session_state.Type == "제품 소개":
        txt = "You are a blog writer who writes an introduction or ad for a product."
    elif st.session_state.Type == "칼럼":
        txt = "You are a blog writer who specializes in writing brief_info."

    if st.session_state.uploaded_file is not None and not st.session_state.get('generated'):

        # Convert the image to Base64
        buffered = BytesIO()
        image = st.session_state.image
        image.save(buffered, format="png")  # Adjust format if necessary (e.g., JPEG)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        with st.spinner('블로그 글 생성중...'):
            st.session_state.blog_content = generate_blog(prompt)
    
    st.session_state['generated'] = True

    # 생성 완료 시
    if st.session_state.get('generated'):
        st.write(st.session_state.blog_content)

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
            with st.spinner('블로그 글 수정중...'):
                refined_content = refine_blog(st.session_state.blog_content, additional_prompt)
                st.session_state.blog_content = refined_content
                st.session_state['refine'] = True
                st.rerun()  # Immediately rerun the script
    with col4:
        if st.button("뒤로 가기"):
            st.session_state.page = "home"
            st.session_state.generated_text = ""
            st.rerun()  # Immediately rerun the script
