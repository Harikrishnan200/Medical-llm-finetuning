FROM python:3.11-slim
WORKDIR /project
RUN useradd --create-home learner
COPY requirements-ci.txt .
RUN pip install --no-cache-dir -r requirements-ci.txt
COPY . .
USER learner
CMD ["python", "-m", "pytest", "-q"]
