import streamlit as st
from src.llm.gpt.inference import run_gpt
from src.llm.deepseek.inference import run_deepseek_stream
from src.search.search import search
from common.types import intlist_struct, str_struct
import asyncio

def get_response(
    prompt: str,
    messages: list,
    name_source_mapping: dict,
    top_k: int = 20,
    refinement_model: str = "gpt-4o-mini",
    reranking_model: str = "gpt-4o-2024-08-06",
    branch: str = "postech",
    collection_name: str = "posplexity-postech-exp"
) -> str:
    """
    RAG + LLM 전체 로직을 처리해 최종 답변 문자열을 반환
    
    1. Query Refinement (gpt-4o-mini)
    2. RAG 검색 (search)
    3. Re-ranking (gpt-4o-2024-08-06)
    4. 최종 RAG 컨텍스트 구성 후 LLM 스트리밍 (deepseek-chat)
    """
    try:
        # 1. Query Refinement
        with st.spinner("질의를 정제 중입니다..."):
            refinement = run_gpt(
                target_prompt=str(prompt),
                prompt_in_path="query_refinement.json",
                gpt_model=refinement_model,
                output_structure=str_struct
            )
        refined_prompt = refinement.output

        # 대화 히스토리 정리
        history_text = ""
        for msg in messages[:-1]:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                history_text += f"User: {content}\n"
            elif role == "assistant":
                history_text += f"Assistant: {content}\n"

        # 2. RAG 검색
        with st.spinner("문서를 조회 중입니다..."):
            found_chunks = search(collection_name, refined_prompt, top_k=top_k, dev=False)

        # 3. Re-ranking
        chunk_dict = {
            c["id"]: (c["doc_title"], c["summary"])
            for c in found_chunks
        }

        with st.spinner("문서를 재정렬 중입니다..."):
            reranked_output = run_gpt(
                target_prompt=str(chunk_dict),
                prompt_in_path="reranking.json",
                gpt_model=reranking_model,
                output_structure=intlist_struct
            )
        reranked_ids = reranked_output.output

        filtered_chunks = [c for c in found_chunks if c["id"] in reranked_ids]
        id_to_rank = {id_: idx for idx, id_ in enumerate(reranked_ids)}
        sorted_chunks = sorted(filtered_chunks, key=lambda x: id_to_rank[x["id"]])

        # 4-1. 최종 RAG 컨텍스트 구성
        context_texts = [c["raw_text"] for c in sorted_chunks]
        rag_context = "\n".join(context_texts)
        final_prompt = f"""
아래는 이전 대화의 기록입니다:
{history_text}

다음은 참고 자료(RAG)에서 발췌한 내용입니다:
{rag_context}

이제 사용자의 질문을 다시 안내해 드리겠습니다:

질문: {refined_prompt}

위 대화와 자료를 기반으로 답변을 작성해 주세요.
답변:
"""

        # 4-2. 최종 답변 (스트리밍)
        return run_final_llm_stream(final_prompt, sorted_chunks, name_source_mapping, branch)

    except Exception as e:
        raise Exception(f"응답 생성 중 오류가 발생했습니다: {str(e)}")


def run_final_llm_stream(final_prompt: str, sorted_chunks: list, name_source_mapping: dict, branch:str) -> str:
    """
    최종 LLM 스트리밍을 수행하고, 완료된 결과를 반환.
    + 출처(Reference) 텍스트도 함께 구성해 반환할 수 있도록 설계 가능
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def _streaming_call():
        message_placeholder = st.empty()
        reference_placeholder = st.empty()

        stream = await run_deepseek_stream(
            target_prompt=final_prompt,
            prompt_in_path="chat_basic.json"
        )

        full_response = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response)

        # 출처 표시
        if sorted_chunks:
            dedup_set = set()
            for c in sorted_chunks:
                doc_title = c.get("doc_title", "Untitled")
                doc_source = c.get("doc_source", "Unknown Source")
                if not doc_source.startswith("http"):
                    doc_source = name_source_mapping[branch].get(doc_title, doc_source)
                page_num = c.get("page_num", None)
                dedup_set.add((doc_title, doc_source, page_num))

            refs = []
            for idx, (title, source, page) in enumerate(dedup_set, start=1):
                if source.startswith("http"):
                    if page is not None:
                        refs.append(f"- **{title}** (p.{page}) / [링크로 이동]({source})")
                    else:
                        refs.append(f"- **{title}** / [링크로 이동]({source})")
                else:
                    if page is not None:
                        refs.append(f"- **{title}** (p.{page}) / {source}")
                    else:
                        refs.append(f"- **{title}** / {source}")

            refs_text = "\n".join(refs)
            reference_placeholder.markdown(
                f"---\n**참고 자료 출처**\n\n{refs_text}\n"
            )

        return full_response

    final_answer = loop.run_until_complete(_streaming_call())
    return final_answer