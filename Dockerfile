FROM python:3.13

WORKDIR /app

COPY ./pyproject.toml /app/pyproject.toml

ENV POETRY_VIRTUALENVS_CREATE=false
RUN pip install poetry
RUN poetry install

COPY . .

CMD ["fastapi", "run", "src/main.py", "--port", "8080"]