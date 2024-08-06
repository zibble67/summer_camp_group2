import streamlit as st
from openai import OpenAI
import time
import pyperclip

client = OpenAI()

if "blog_content" not in st.session_state:
    st.session_state.blog_content = ""

def generate_blog(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a blog writer."},
            {"role": "user", "content": f"Please write a blog writing according to the following information: {prompt}"}
        ],
        max_tokens=300,
        stream=False
    )

    return response.choices[0].message.content

def refine_blog(blog_content, additional_prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Modify the blog according to t"},
            {"role": "assistant", "content": blog_content},
            {"role": "user", "content": additional_prompt}
        ],
        max_tokens=300,
        stream=True
    )

    return response


def main():
    st.title("블로그 게시글 생성기")

    # 임시 입력창
    blog_prompt = st.text_area("원하는 블로그 내용을 입력하세요:", "")

    if st.button("글 생성하기"):
        if blog_prompt.strip() == "":
            st.warning("생성할 글에 대한 정보를 입력해주세요.")
        else:
            with st.spinner('블로그 글 생성중...'):
                blog_content = generate_blog(blog_prompt)
                st.session_state['blog_content'] = blog_content
                st.session_state['generated'] = True

    # 생성 완료 시
    if st.session_state.get('generated'):
        blog_content = st.session_state['blog_content']
        st.write(blog_content)

    # 추가 수정 프롬프팅
    additional_prompt = st.text_area("수정을 원한다면 추가 정보를 입력하세요.")

    if st.button("수정하기"):
        with st.spinner('블로그 글 수정중...'):
            refined_content = refine_blog(st.session_state.blog_content, additional_prompt)
            st.session_state['blog_content'] = refined_content

if __name__ == "__main__":
    st.set_page_config(page_title="blogwriter")
    if 'generated' not in st.session_state:
        st.session_state['generated'] = False
    main()