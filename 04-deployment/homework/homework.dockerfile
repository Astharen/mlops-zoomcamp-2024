FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim


COPY ["requirements.txt", "./"]

RUN pip install -r requirements.txt

COPY ["starter.py", "starter.py"]

ENTRYPOINT ["python", "starter.py"]
 