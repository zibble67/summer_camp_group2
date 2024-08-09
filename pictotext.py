import streamlit as st
import base64
from io import BytesIO
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import pyperclip
from datetime import datetime, timedelta
from geopy.distance import geodesic
import os
from datasets import load_dataset 
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI as llOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import nest_asyncio
import pandas as pd
from openai import OpenAI

openai_api_key = ""
client = OpenAI(api_key=openai_api_key)

# Streamlit UI
st.title("블로그 게시글 생성기")

################### 변수 초기화 & 상수 정의 ########################
# 1분 이내와 근접한 위치를 정의하는 상수
TIME_THRESHOLD = timedelta(minutes=120)
DISTANCE_THRESHOLD = 100  # 거리 임계값 (단위: km, 여기서는 100m)

if "page" not in st.session_state:  # 웹 페이지 태그
    st.session_state.page = "home"
if "blog_content" not in st.session_state:  # 블로그 작성글 저장할 컨테이너
    st.session_state.blog_content = []
if "Type" not in st.session_state:  # 블로그 유형 태그
    st.session_state.Type = ""
if 'generated' not in st.session_state: # 글 생성 플래그
    st.session_state['generated'] = False
if 'info' not in st.session_state:  # 사용자 지정 정보
    st.session_state.info = {}
if 'refine' not in st.session_state:    # 글 수정 플래그
    st.session_state['refine'] = False
if "group_info" not in st.session_state:    # 사진 그룹 정보
    st.session_state.group_info = []

################## 필요한 함수 정의 #####################
def generate_blog(prev_content, prompt, photo):      # 블로그 글 작성 함수
    filtered_dict = {k: v for k, v in photo.items() if k != 'img_base64'}

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"{prompt}, {prev_content} (Refer to the prev_content, but avoid writing overlapping with this content and please write it so that it connects with the prev_content)"},
            {"role": "user", "content":[
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{photo['img_base64']}",
                    },
                    },
                {"type": "text", "text": f"Please write a blog writing in korean according to images and the following information: {filtered_dict}."}, 
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
            {"role": "user", "content": f"Keep the contents of the existing text(blog_content) as much as possible, but modify it appropriately by reflecting {additional_prompt}, the information given by the user."}
        ],
        #max_tokens=1000,
        stream=False
    )

    return response.choices[0].message.content

def get_exif_data(image):   # 이미지 메타데이터 추출 함수(반환: 딕셔너리)
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

def correct_image_orientation(image, exif_data):    # 이미지 회전 함수(반환: 회전된 이미지)
    if 'Orientation' in exif_data:
        orientation = exif_data['Orientation']
        if orientation == 3:
            image = image.rotate(180, expand=True)
        elif orientation == 6:
            image = image.rotate(270, expand=True)
        elif orientation == 8:
            image = image.rotate(90, expand=True)
    return image

def extract_datetime(exif_data):    # 이미지 시간 데이터 추출 함수(반환: datetime)
    if 'DateTimeOriginal' in exif_data:
        date_time_str = exif_data['DateTimeOriginal']
        try:
            return datetime.strptime(date_time_str, "%Y:%m:%d %H:%M:%S")
        except ValueError:
            return None
    return None

def get_gps_info(exif_data):    # 사진 속 장소 gps 데이터 얻는 함수(반환: 딕셔너리)
    gps_info = {}
    if 'GPSInfo' in exif_data:
        for key in exif_data['GPSInfo'].keys():
            decode = GPSTAGS.get(key, key)
            gps_info[decode] = exif_data['GPSInfo'][key]
    return gps_info

def get_coordinates(gps_info):  # 사진 속 장소 위도/경도 데이터 얻는 함수(반환: 실수? 문자열?)
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

def group_photos_by_time_and_location(photos_info): # 시간과 장소로 사진 그룹화 함수(반환: '딕셔너리 배열을 담고 있는 리스트'를 원소로 갖는 리스트)
    groups = []
    for photo in photos_info:
        matched = False
        for group in groups:
            time_diff = abs(group[-1]['datetime'] - photo['datetime'])
            if time_diff <= TIME_THRESHOLD: # 어떤 그룹의 마지막 사진과 타겟 사진의 시간차가 임계점 이하이면서
                if photo['lat'] is not None and photo['lon'] is not None:
                    last_photo = group[-1]
                    distance = geodesic((last_photo['lat'], last_photo['lon']), (photo['lat'], photo['lon'])).km
                    if distance <= DISTANCE_THRESHOLD:  # 그룹의 마지막 사진과 타겟 사진의 위치차가 임계점 이하이면
                        group.append(photo) # 그룹화
                        matched = True
                        break
        if not matched:
            groups.append([photo])
    return groups

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

######################### 시작 UI ############################
if st.session_state.page == "home":

    Type = st.radio(
        "원하는 옵션을 선택하세요.",
        ["일상 기록", "제품 소개", "칼럼", "검색"],
        captions=[
            "기록하고 싶은 일상.",
            "편리한 광고 작성.",
            "전문적인 칼럼.",
            "당신만의 고유한 아카이브."
        ], 
    )
    st.session_state.Type = Type

    if Type in ["일상 기록", "제품 소개", "칼럼"]:
        uploaded_files = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if st.button("생성"):
            st.session_state.page = "result"
            st.session_state['refine'] = False
            st.rerun()  # Immediately rerun the script

        if Type == "일상 기록":
            mood = st.selectbox("Mood", ["활기찬", "우울한", "무던한"])
            st.session_state.info['mood'] = mood
        tone = st.selectbox("말투", ["격식있는", "캐주얼한", "유머러스한", "CUSTOM"])
        st.session_state.info['tone'] = tone
        st.write("---")  # 구분선 추가

        if uploaded_files is not None:
            photos_info = []    # 딕셔너리 배열을 담고있는 리스트

            for uploaded_file in uploaded_files:
                
                # Display the uploaded image
                image = Image.open(uploaded_file)
                exif_data = get_exif_data(image)

                # 이미지 회전 보정
                image = correct_image_orientation(image, exif_data)
                st.session_state.image = image
                
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
            grouped_photos = group_photos_by_time_and_location(photos_info) # 딕셔너리 배열을 담고있는 리스트의 리스트 [[{}, {}, ...], [...], ... ]

            st.session_state.group_info = []
            # 그룹화된 사진 출력 (같은 행에 배치)
            for i, group in enumerate(grouped_photos):  # group == 딕셔너리 배열을 담고있는 리스트 (같은 그룹의 사진들의 정보들의 리스트)
                st.subheader(f"그룹 {i+1}")
                cols = st.columns(len(group))  # 그룹 내 사진 수에 따라 컬럼 생성
                for idx, info in enumerate(group):  # info == 어떤 사진의 정보들(딕셔너리)
                    with cols[idx]:
                        st.image(info["image"], caption=f"{info['filename']} (촬영 시간: {info['datetime']})", use_column_width=True)
                        if not info["lat"] or not info["lon"]:
                            st.write("위치 정보를 찾을 수 없습니다.")
                    text = st.text_area(f"{info['filename']}에 대한 메모를 입력하세요.")
                    info['text'] = text
                st.session_state.group_info.append(group)
                st.write("---")  # 구분선 추가


    elif Type == "검색":
        ################## llamaindex 세팅 #######################
        nest_asyncio.apply()
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # RAG 파이프라인 글로벌 설정
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small"
        )

        Settings.llm=llOpenAI(model='gpt-3.5-turbo',temperature=0)

        # CSV 파일 로드
        df = pd.read_csv("C:/Users/pizzazoa/Downloads/separated_travel_data.csv")

        # 데이터프레임을 라마인덱스 다큐먼트 객체로 변환
        docs = []
        for i, row in df.iterrows():
            docs.append(Document(
                text=row['Content'],
                extra_info={'date': row['Date'], 'place': row['Place']}
            ))

        # RAG 파이프라인 글로벌 설정
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small"
        )

        Settings.llm = llOpenAI(model='gpt-4o-mini', temperature=0)

        # 벡터스토어 인덱스 설정
        vector_index = VectorStoreIndex.from_documents(
            docs,
            use_asnyc=True
        )

        # 쿼리 엔진 설정
        vector_query_engine = vector_index.as_query_engine(similarity_top_k=2)

        # 질문하기 버튼
        question_prompt = st.chat_input('추억을 검색해보세요!')

        answer_num = 1

        if question_prompt:
            response = vector_query_engine.query(question_prompt)
            st.write(f'다음은 검색 결과 입니다: {response.response}')
            for node in response.source_nodes:
                for node in response.source_nodes:
                    st.write(f'''
        {answer_num}번 검색 결과 : 
        - 유사도 점수: {node.score}, 
        - 자료 속 긁어온 텍스트: {node.node.text}''')
                    answer_num += 1


#################### 중간 UI ########################
elif st.session_state.page == "result":

    if st.session_state.Type == "일상 기록":
        prompt = f'''너는 인기있는 여행블로거의 역할을 맡을꺼야. 블로그 글을 써야하는데, 아래 조건과 추가 명령 사항을 참고해서, 입력된 사진에 알맞는 글을 작성해줘. 
        그 글은, 여행 다녀온 후기에 맞는 글이어야 해. 지금이 5월이라고 생각하고 화이팅해서 작성해줘 고마워. 
        가장 중요한 것은, 예시를 참고하여 형식적인 GPT 글이 아닌, 사람의 속마음이 담겨져서 사람이 작성한 글처럼 보여야해.
        아, 그리고 이모지(emoji)를 중간중간에 3개 가량 꼭 넣어줘. 다른 설명은 할 필요 없고, 블로그 글만 써줘!
        글 작성을 완료하기전에, 예시1 과 예시2와 형식이 비슷한지 꼭 확인해줘. 저 예시들과 비슷하게 나왔으면 좋겠어

        # 조건
        - 당시 기분은 {st.session_state.info["mood"]}
        - 여행 블로그 글의 어투는 {st.session_state.info["tone"]}으로
        - 반말을 기본으로
        - 글자수는 사진 1장당 약 300자 에서 500자 사이
        - 중요한 정보는 글자색 다르게 혹은 굵기를 진하게 하여 표시
        - 구체적인 수치를 기본으로
        - 한문단에 2줄 넘게 작성되면 안됨
        
        # 추가명령
        - 사진이 어떤 사진인지 파악하여 그에 알맞는 글을 생성해낸다.
        (만약 음식 사진이라면, {st.session_state.info["mood"]}를 참고하여 음식의 맛에 대한 평가가 담긴 글이어야함)
        - 마지막에 간단한 한마디를 추가하기

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
        prompt = "You are a blog writer who writes an introduction or ad for a product."
    elif st.session_state.Type == "칼럼":
        prompt = "You are a blog writer who specializes in writing brief_info."

    if st.session_state.group_info is not None and not st.session_state.get('generated'):
        
        with st.spinner('블로그 글 생성중...'):
            for group in st.session_state.group_info:
                prev_content = ''
                for photo in group:
                    blog_content = generate_blog(prev_content, prompt, photo)
                    st.session_state.blog_content.append({'content': blog_content, 'image': photo['image']})
                    prev_content = blog_content
    
    st.session_state['generated'] = True

    # 생성 완료 시
    if st.session_state.get('generated'):
        for content in st.session_state.blog_content:
            st.image(content['image'])
            st.write(content['content'])

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
                st.session_state.refined_content = []
                for content in st.session_state.blog_content:
                    refined_content = refine_blog(content['content'], additional_prompt)
                    st.session_state.refined_content.append({'content': refined_content, 'image': content['image']})
                st.session_state.blog_content = st.session_state.refined_content
                st.session_state['refine'] = True
                st.rerun()  # Immediately rerun the script
    with col4:
        if st.button("뒤로 가기"):
            st.session_state.page = "home"
            # 초기화
            st.session_state.blog_content = []
            st.session_state['refine'] = False
            st.session_state.Type = ""
            st.session_state['generated'] = False
            st.session_state.info = {}
            st.session_state.group_info = []
            st.rerun()  # Immediately rerun the script
