FROM public.ecr.aws/lambda/python:3.13

COPY requirements.txt ${LAMBDA_TASK_ROOT} 

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN mkdir ${LAMBDA_TASK_ROOT}/my_local_model && \
    pip install sentence-transformers -t ${LAMBDA_TASK_ROOT}/my_local_model

RUN python -W ignore -m nltk.downloader punkt

RUN mv /root/nltk_data /usr/local/share/nltk_data

COPY nltk_data ${LAMBDA_TASK_ROOT}/nltk_data

ENV PYTHONPATH="${PYTHONPATH}:${LAMBDA_TASK_ROOT}/my_local_model"

COPY . ${LAMBDA_TASK_ROOT} 

CMD [ "lambda_function.lambda_handler" ]