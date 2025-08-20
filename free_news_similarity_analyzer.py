import json
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util

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

# 메인 처리 함수
def free_analyze_and_summarize(dto, threshold=0.5):
    main_news = dto.get("mainNewsDto", {})
    main_sentences = _to_sentences((main_news.get("content") or ""))

    # 메인 본문 자체가 비었을 경우 모든 유사도를 0.0으로 반환
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
                }
            })
        comparison_results.sort(key=lambda x: x["newsWithSimilarityDto"]["similarity"], reverse=True)
        result = {
            "category": dto.get("category", "free"),
            "mainNewsDto": main_news,
            "newsComparisonDtos": comparison_results
        }
        return json.dumps(result, ensure_ascii=False)

    main_embeddings = model.encode(main_sentences, convert_to_tensor=True)

    comparison_results = []
    for article in dto.get("newsDtos", []):
        article_sentences = _to_sentences((article.get("content") or ""))   # [변경] sent_tokenize 대신 _to_sentences
        similarity = _similarity_from_sentences(main_embeddings, article_sentences)  # [변경] 직접 유사도 계산 대신 헬퍼 호출

        comparison_results.append({
            "newsWithSimilarityDto": {
                "similarity": similarity,
                "newsDto": {
                    "url": article.get("url", ""),
                    "title": article.get("title", ""),
                    "content": article.get("content", "")
                }
            }
        })

    comparison_results.sort(key=lambda x: x["newsWithSimilarityDto"]["similarity"], reverse=True)

    result = {
        "category": dto.get("category", "free"),
        "mainNewsDto": main_news,
        "newsComparisonDtos": comparison_results
    }
    return json.dumps(result, ensure_ascii=False)