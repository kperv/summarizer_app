FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY . /app

EXPOSE 8000

CMD ["Summarizer:app", "--host", "0.0.0.0", "--port", "8000"]

