#FROM ubuntu:16.04
FROM python:3.8.8

ARG PACKAGE_FILE
RUN apt-get update && apt-get install -y \
    wget \
    nginx \
    ca-certificates \
    zip \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
ENV NLTK_DATA /opt/nltk_data

COPY requirements.txt ./
COPY ${PACKAGE_FILE} ./


RUN wget https://github.com/nltk/nltk_data/raw/gh-pages/packages/corpora/stopwords.zip \
    && wget https://github.com/nltk/nltk_data/raw/gh-pages/packages/tokenizers/punkt.zip \
    && wget https://github.com/nltk/nltk_data/raw/gh-pages/packages/corpora/wordnet.zip \
    && unzip stopwords.zip \
    && unzip punkt.zip \
    && unzip wordnet.zip \
    && mkdir -p $NLTK_DATA/corpora \
    && mkdir -p $NLTK_DATA/tokenizers \
    && mv stopwords/ $NLTK_DATA/corpora/stopwords \
    && mv wordnet/ $NLTK_DATA/corpora/wordnet \
    && mv punkt/ $NLTK_DATA/tokenizers/punkt


RUN python -m pip install --upgrade pip
RUN pip install ${PACKAGE_FILE} --no-cache-dir --upgrade \
    && rm ${PACKAGE_FILE}
RUN pip install -r requirements.txt --no-cache-dir


COPY api /opt/program/api
COPY run.py /opt/program/
WORKDIR /opt/program
EXPOSE 8080
ENTRYPOINT ["python", "run.py"]