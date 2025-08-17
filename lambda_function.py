import json

from free_news_similarity_analyzer import free_analyze_and_summarize
from premium_news_similarity_analyzer import premium_analyze_and_summarize

def lambda_handler(event, context):
    try:
        body_str = json.dumps(event)
        body = json.loads(body_str)

        plan = body.get("plan", "free")  # 기본: free
        if plan == "premium":
            result = premium_analyze_and_summarize(body)
        else:
            result = free_analyze_and_summarize(body)

        return {
            'statusCode': 200,
            'body': result
        }
    except Exception as e:
        import traceback
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "trace": traceback.format_exc()
            })
        }
