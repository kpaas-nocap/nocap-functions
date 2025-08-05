import json
from free_news_similarity_analyzer import free_analyze_and_summarize
from premium_news_similarity_analyzer import premium_analyze_and_summarize

def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])

        mode = body.get("mode", "free")  # 기본: free
        if mode == "premium":
            result = premium_analyze_and_summarize(body)
        else:
            result = free_analyze_and_summarize(body)

        return {
            'statusCode': 200,
            'body': result
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }
