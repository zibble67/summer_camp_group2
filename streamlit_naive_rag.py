# 관련 패키지 임포트
import os
from datasets import load_dataset 
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import nest_asyncio
import pandas as pd
import re
import streamlit as st


nest_asyncio.apply()

# 활용 LLM API 정보 설정
os.environ["OPENAI_API_KEY"] = ''

# RAG 파이프라인 글로벌 설정
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)

Settings.llm=OpenAI(model='gpt-3.5-turbo',temperature=0)


# CSV 파일 로드
df = pd.read_csv('Your-file-path')

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

Settings.llm = OpenAI(model='gpt-4o-mini', temperature=0)

# 벡터스토어 인덱스 설정
vector_index = VectorStoreIndex.from_documents(
    docs,
    use_asnyc=True
)

# 쿼리 엔진 설정
vector_query_engine = vector_index.as_query_engine(similarity_top_k=2)

# # 질문을 통해 검색
# question = '그때 갔던 그 카페 이름이 뭐지?'
# response = vector_query_engine.query(question)

# # 검색에 대한 답변 출력
# print("답변:", response.response)

# # 소스 노드 출력
# for node in response.source_nodes:
#     print(f"점수: {node.score}, 텍스트: {node.node.text}")


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

    