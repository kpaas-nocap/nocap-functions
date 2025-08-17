FROM public.ecr.aws/lambda/python:3.13

COPY requirements.txt ${LAMBDA_TASK_ROOT} 

RUN pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt

RUN python -m nltk.downloader punkt

COPY . ${LAMBDA_TASK_ROOT} 

CMD [ "lambda_function.lambda_handler" ]