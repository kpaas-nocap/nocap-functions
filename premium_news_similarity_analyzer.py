import json
import os
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

client = os.getenv('OPENAI_API_KEY')

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

_ = model.encode(["프리로딩 테스트"], convert_to_tensor=True)

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
    main_news = dto["mainNews"]
    main_sentences = sent_tokenize(main_news["fullContent"])
    main_embeddings = model.encode(main_sentences, convert_to_tensor=True)

    comparison_results = []

    for article in dto["newsItem"]:
        article_sentences = sent_tokenize(article["fullContent"])
        article_embeddings = model.encode(article_sentences, convert_to_tensor=True)

        sim_matrix = util.cos_sim(main_embeddings, article_embeddings)
        similarity = round(sim_matrix.max().item(), 4)

        # GPT 비교 요약은 유사도 0.5 이상일 때만 실행
        comparison_text = None
        if similarity >= threshold:
            try:
                comparison_text = generate_comparative_summary(
                    main_news["fullContent"], article["fullContent"]
                )
            except Exception as e:
                comparison_text = f"요약 실패: {str(e)}"

        # 결과 구성
        comparison_results.append({
            "newsWithSimilarityDTO": {
                "similarity": similarity,
                "news": {
                    "url": article["url"],
                    "title": article.get("title", ""),
                    "fullContent": article["fullContent"]
                }
            },
            "comparision": comparison_text
        })

    comparison_results.sort(key=lambda x: x["newsWithSimilarityDTO"]["similarity"], reverse=True)

    result = {
        "category": dto.get("category", ""),
        "mainNews": main_news,
        "newsComparisionDTOS": comparison_results
    }
    return json.dumps(result, ensure_ascii=False)