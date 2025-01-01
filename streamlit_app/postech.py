import sys, os, asyncio, json
sys.path.append(os.path.abspath("")) 

import streamlit as st
from common.config import POSTECH_COLLECTION_PROD, POSTECH_COLLECTION_EXP
from core import get_response

try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

name_source_mapping = json.load(open("data/mapping.json", "r"))

def setup_sidebar():
    """사이드바 UI 구성"""

    # Posplexity logo
    st.sidebar.image(
        "data/assets/posplexity_light.png",
        use_container_width=True
    )
    
    st.sidebar.markdown("""
    \n새내기 여러분의 궁금증을 해소하기 위해 관련 자료를 기반으로 답변을 제공하는 챗봇입니다.
    """)

    # 예시 질문 
    with st.sidebar.expander("ℹ️ 예시 질문", expanded=True):
        example_questions = [
            "밥약이 무슨 뜻인가요?",
            "새터 기간동안 술을 마셔도 괜찮나요?",
            "야구를 좋아하는데, 어떤 동아리에 들어가는게 좋을까요?",
        ]
        for question in example_questions:
            if st.button(question):
                st.session_state.pending_question = question
                st.rerun()

    st.sidebar.divider()

    with st.sidebar.expander("💬 문의하기", expanded=False):
        st.markdown("""                    
            ### Contact
            궁금한 점이나 피드백은 언제든지 아래 페이지를 통해 공유해 주세요.
            - [문의사항 페이지](https://forms.gle/aMAJA7yPFfCRGLro9)
                    
            ### Contributing
            자료를 보완하거나 새롭게 추가하고 싶은 내용이 있다면, 아래 업로드 페이지를 이용해 주시기 바랍니다.
            - [업로드 페이지](https://docs.google.com/forms/d/e/1FAIpQLScUW14gj69mWXlhoKpJejBLWCbj-wOQZ4e6XQT69ZFNWZS4SA/viewform)
        """)

    with st.sidebar.expander("👨‍👩‍👦‍👦 제작자", expanded=False):
        st.markdown("""
            ### Contributors
            [**허채원**](https://www.linkedin.com/in/cwhuh/)(포스텍 24),  
            [**최지안**](https://www.linkedin.com/in/%EC%A7%80%EC%95%88-%EC%B5%9C-72093030a/)(포스텍 24),  
            [**최주연**](https://www.linkedin.com/in/%EC%A3%BC%EC%97%B0-%EC%B5%9C-a9884331b/)(포스텍 24),  
            [**정찬희**](https://www.linkedin.com/in/%EC%B0%AC%ED%9D%AC-%EC%A0%95-b6506b328/)(포스텍 24)
        """)

    # Postech logo
    # st.sidebar.image(
    #     "data/assets/postech/postech_logo.svg",
    #     width=150
    # )


def setup_page():
    """메인 페이지(본문) 설정을 담당. 타이틀, 부가 문구 등을 표시."""

    st.title("포스텍 2025 입학을 축하합니다!")
    st.caption("powered by P13")


# Streamlit Settings
st.set_page_config(page_title="Posplexity", layout="wide")

setup_sidebar()
setup_page()

# Default messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "안녕하세요! 저는 POSTECH 새내기 여러분을 도와드리는 챗봇입니다.\n"
                "무엇이든 궁금한 점이 있다면 편하게 물어보세요."
            )
        }
    ]

# 과거 대화 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 예시 질문 처리 / user_input
prompt = None
if "pending_question" in st.session_state:
    prompt = st.session_state.pending_question
    del st.session_state.pending_question

user_input = st.chat_input("질문을 입력하세요")
if user_input:
    prompt = user_input

if prompt:
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 코어 로직 호출
    with st.chat_message("assistant"):
        try:
            final_response = get_response(
                prompt=prompt,
                messages=st.session_state.messages,
                name_source_mapping=name_source_mapping,
                branch="postech",
                collection_name=POSTECH_COLLECTION_PROD
            )
            # 최종 응답을 세션 메시지에 저장
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_response
            })
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")