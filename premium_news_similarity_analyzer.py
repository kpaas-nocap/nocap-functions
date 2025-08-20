import json
import os
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model = SentenceTransformer("./model")

_ = model.encode(["프리로딩 테스트"], convert_to_tensor=True)

# 빈값/공백/None 방어 함수
def _to_sentences(text: str):
    if not isinstance(text, str):
        return []
    
    # 제로폭공백 등 제거 후 트림
    cleaned = text.replace("\u200b", "").strip()
    if not cleaned:
        return []
    
    # 공백 문장 제거
    return [s.strip() for s in sent_tokenize(cleaned) if s.strip()]

# 기사 본문 비었을 경우 유사도를 0.0으로 처리
def _similarity_from_sentences(main_embeddings, article_sentences):
    # 기사 문장이 없을 경우
    if not article_sentences:
        return 0.0

    article_embeddings = model.encode(article_sentences, convert_to_tensor=True)

    # 임베딩이 빈 텐서가 되면 0.0
    if getattr(article_embeddings, "shape", [0])[0] == 0 or getattr(main_embeddings, "shape", [0])[0] == 0:
        return 0.0

    sim_matrix = util.cos_sim(main_embeddings, article_embeddings)

    # 빈 행렬 보호
    if sim_matrix.numel() == 0:
        return 0.0

    return round(sim_matrix.max().item(), 4)

# 유사도 점수 + GPT 요약
def generate_comparative_summary(main_content, compare_content):
    prompt = (
        "당신은 뉴스 기사 비교 전문가입니다.\n\n"
        "메인 기사:\n" + main_content[:1000] + "\n\n"
        "비교 기사:\n" + compare_content[:1000] + "\n\n"
        "메인 기사와 비교 기사 간 핵심 유사점과 차이점을 한국어로 한두 문장으로 간결하게 설명해주세요." +
        "예시: “메인 기사에서는 A를 k라고 표현하지만, 비교 기사에서는 이를 q로 다루고 있습니다."
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "너는 뉴스 비교 요약 전문가야."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()

# 메인 처리 함수
def premium_analyze_and_summarize(dto, threshold=0.5):
    main_news = dto.get("mainNewsDto", {})
    main_sentences = _to_sentences((main_news.get("content") or ""))
    
    if not main_sentences:
        comparison_results = []
        for article in dto.get("newsDtos", []):
            comparison_results.append({
                "newsWithSimilarityDto": {
                    "similarity": 0.0,
                    "newsDto": {
                        "url": article.get("url", ""),
                        "title": article.get("title", ""),
                        "content": article.get("content", "")
                    }
                },
                "comparison": None  
            })
        comparison_results.sort(key=lambda x: x["newsWithSimilarityDto"]["similarity"], reverse=True)
        result = {
            "category": dto.get("category", ""),
            "mainNewsDto": main_news,
            "newsComparisonDtos": comparison_results
        }
        return json.dumps(result, ensure_ascii=False)

    main_embeddings = model.encode(main_sentences, convert_to_tensor=True)

    comparison_results = []

    for article in dto.get("newsDtos", []):
        article_sentences = _to_sentences((article.get("content") or ""))   # [변경]
        similarity = _similarity_from_sentences(main_embeddings, article_sentences)  # [변경]

        # GPT 비교 요약은 유사도 threshold 이상일 때만 실행
        comparison_text = None
        if similarity >= threshold:
            try:
                comparison_text = generate_comparative_summary(
                    main_news.get("content", ""), article.get("content", "")
                )
            except Exception as e:
                comparison_text = f"요약 실패: {str(e)}"

        comparison_results.append({
            "newsWithSimilarityDto": {
                "similarity": similarity,
                "newsDto": {
                    "url": article.get("url", ""),
                    "title": article.get("title", ""),
                    "content": article.get("content", "")
                }
            },
            "comparison": comparison_text
        })

    comparison_results.sort(key=lambda x: x["newsWithSimilarityDto"]["similarity"], reverse=True)

    result = {
        "category": dto.get("category", ""),
        "mainNewsDto": main_news,
        "newsComparisonDtos": comparison_results
    }
    return json.dumps(result, ensure_ascii=False)