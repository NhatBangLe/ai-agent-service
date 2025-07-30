# AI Agent Service

AI Agent Service is a part of Creating AI Agent Framework from my bachelor's graduation thesis.

Creating AI Agent Framework includes two parts:
- AI Agent Configuration Toolkit.
- AI Agent Service.

## Installation

Use the Poetry package manager [poetry](https://python-poetry.org/docs/basic-usage/#installing-dependencies) to install dependencies.

At the folder that contains a `pyproject.toml` file, use this command:
```bash
poetry install --no-root
```

We need configuration files that are created by using AI Agent Configuration Toolkit. Put the configuration files at anywhere you want, you just need configuring a path to them.

Required environment variables:
- `POSTGRES_HOST`: PostgreSQL database host.
- `POSTGRES_PORT`: PostgreSQL database port.
- `POSTGRES_DATABASE`: PostgreSQL database name.
- `POSTGRES_USER`: PostgreSQL database username.
- `POSTGRES_PASSWORD`: PostgreSQL database password.
- `AGENT_CONFIG_DIR`: Path to folder containing the created configuration folder.
- `API_KEY`: A LLM API key to use. The variable name depends on what your LLM provider you use.

Additional environment variables (optional):
- `LOG_LEVEL`: Logging level, default `INFO`. Available values: INFO, WARNING, DEBUG.
- `CACHE_DIR`= Path to a cache folder, default in `/rag_agent/cache` folder.
- `SAVE_FILE_DIR`: Path to local saved files folder, default in `/rag_agent/resource` folder.
- `MAX_WORKERS`: A value to initialize thread pool for using CNN model purpose, default `4`.
- `DOWNLOAD_GENERATOR_SECRET_KEY`: A secret key to generate tokens for downloading files, default `your-super-secret-key-change-in-production`.

## Usage

1. Run PostgreSQL database service.
2. Run the server, use the below command:
```bash
fastapi run ./src/main.py
```

Check the API documentation at `/docs` to see endpoints.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)