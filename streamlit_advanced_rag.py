# 관련 패키지 임포트
import os
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
# from llama_index.postprocessor.cohere_rerank import CohereRerank


nest_asyncio.apply()

# 활용 LLM API 정보 설정
os.environ["OPENAI_API_KEY"] = ''

# RAG 파이프라인 글로벌 설정
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)

Settings.llm=OpenAI(model='gpt-4o-mini',temperature=0)


# CSV 파일 로드
df = pd.read_csv('/Users/zibble/Desktop/separated_travel_data.csv')

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

Settings.llm = OpenAI(model='gpt-4o-mini', temperature = 0)

# 벡터스토어 인덱스 설정
vector_index = VectorStoreIndex.from_documents(
    docs,
    use_asnyc=True
)

# 쿼리 엔진 설정
naive_vector_query_engine = vector_index.as_query_engine(similarity_top_k = 2)



# Advanced RAG 1 (Pre-retrival 단계) : Sub-question query engine 설정
Subquestion_query_engine_tools = [
    QueryEngineTool(
        query_engine = naive_vector_query_engine,
        metadata = ToolMetadata(
            name = "blog post",
            description="사용자 질문이 너무 길때 쪼개서 입력 받고 에 대답하기",
        ),
    ),
]

advanced_query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=Subquestion_query_engine_tools,
    use_async=True,
)

# # Advanced RAG 2 (Post-retrival 단계) : Re-rank 단계 추가

# cohere_api_key = ''
# cohere_rerank = CohereRerank(api_key=cohere_api_key, top_n=2)

# rerank_query_engine = vector_index.as_query_engine(
#     similarity_top_k = 10,
#     node_postprocessors = [cohere_rerank],
# )


# 질문하기 버튼
question_prompt = st.chat_input('추억을 검색해보세요!')

answer_num = 1

if question_prompt:
    response = advanced_query_engine.query(question_prompt)
    # response = rerank_query_engine.query(question_prompt)
    st.write(f'다음은 검색 결과 입니다: {response.response}')
    for node in response.source_nodes:
        for node in response.source_nodes:
            st.write(f'''
{answer_num}번 검색 결과 : 
- 유사도 점수: {node.score}, 
- 자료 속 긁어온 텍스트: {node.node.text}''')
            answer_num += 1

    