FROM public.ecr.aws/lambda/python:3.13

COPY requirements.txt ${LAMBDA_TASK_ROOT} 

RUN pip install --upgrade pip \
 && pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY . ${LAMBDA_TASK_ROOT} 

CMD [ "lambda_function.lambda_handler" ]