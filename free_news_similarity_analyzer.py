import json
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("./model")

_ = model.encode(["프리로딩 테스트"], convert_to_tensor=True)

# 메인 처리 함수
def free_analyze_and_summarize(dto, threshold=0.5):
    main_news = dto["mainNewsDto"]
    main_sentences = sent_tokenize(main_news["content"])
    main_embeddings = model.encode(main_sentences, convert_to_tensor=True)

    comparison_results = []
    for article in dto["newsDtos"]:
        article_sentences = sent_tokenize(article["content"])
        article_embeddings = model.encode(article_sentences, convert_to_tensor=True)

        sim_matrix = util.cos_sim(main_embeddings, article_embeddings)
        similarity = round(sim_matrix.max().item(), 4)

        # 결과
        comparison_results.append({
            "newsWithSimilarityDto": {
                "similarity": similarity,
                "newsDto": {
                    "url": article["url"],
                    "title": article.get("title", ""),
                    "content": article["content"]
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