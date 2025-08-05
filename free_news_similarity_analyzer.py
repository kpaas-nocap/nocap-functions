import json
import os
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

nltk.download('punkt')

load_dotenv('.env')
client = os.getenv('OPENAI_API_KEY')

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 메인 처리 함수
def free_analyze_and_summarize(dto, threshold=0.5):
    main_news = dto["mainNews"]
    main_sentences = sent_tokenize(main_news["fullContent"])
    main_embeddings = model.encode(main_sentences, convert_to_tensor=True)

    comparison_results = []

    for article in dto["newsItem"]:
        article_sentences = sent_tokenize(article["fullContent"])
        article_embeddings = model.encode(article_sentences, convert_to_tensor=True)

        sim_matrix = util.cos_sim(main_embeddings, article_embeddings)
        similarity = round(sim_matrix.max().item(), 4)

        # 결과 구성
        comparison_results.append({
            "newsWithSimilarityDTO": {
                "similarity": similarity,
                "news": {
                    "url": article["url"],
                    "title": article.get("title", ""),
                    "fullContent": article["fullContent"]
                }
            }
        })

        comparison_results.sort(key=lambda x: x["newsWithSimilarityDTO"]["similarity"], reverse=True)

    result = {
        "category": dto.get("category", ""),
        "mainNews": main_news,
        "newsComparisionDTOS": comparison_results
    }
    return json.dumps(result, ensure_ascii=False)