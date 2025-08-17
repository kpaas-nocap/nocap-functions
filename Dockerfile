FROM public.ecr.aws/lambda/python:3.13

COPY requirements.txt ${LAMBDA_TASK_ROOT} 
 
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt


# NLTK punkt 다운로드
RUN python -m nltk.downloader punkt -d /usr/share/nltk_data

RUN python -m nltk.downloader punkt_tab -d /usr/share/nltk_data

# Hugging Face 모델 미리 다운로드
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

COPY . ${LAMBDA_TASK_ROOT}

CMD [ "lambda_function.lambda_handler" ]