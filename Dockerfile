FROM python:3
ENV PYTHONUNBUFFERED 1

WORKDIR /opt/TheSignalProject/
COPY ./ ./

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["./cron.sh"]
