# retrospective-api-base
Get past conversation data, summarize it, and create retrospective

## Installation

> docker or docker desktop should be installed on your system

### 1. Prepare volumes

#### 1-1. Create volume for saving local models (`.gguf` models)

```bash
docker volume model_data
```

- name of volume should be exactly `model_data` unless you change `docker-compose.yml`
- assume that you have already prepared the `.gguf` model which will be referenced inside of app via `llama-cpp-python`

#### 1-2. Use temporary container to move the model file from your system to created volume

run temporary linux container with mounting the volume 

```bash
docker run -dit --name temp-container -v model_data:/app/model ubuntu:latest
```

- use simple ubuntu image
- `-d` : run container on background (not interactive shell)

copy local model to inside of volume's path

```bash
docker cp /path/to/your/model.gguf temp-container:/app/model
```

stop and delete container

```bash
docker stop temp-container
docker rm -f temp-container
```

now the local model is ready

#### 1-3. Create volume for postgres database

create dedicated volume for postgres db

```bash
docker volume postgres_data
```

### 2. Setup environment variables

create `.env` file from `.env.template` and fill out empty string as you need

> it is possible to use double quotes ("") but, conventionally, no quotes are used in environment file.

```.env
DB_USERNAME=?
DB_PASSWORD=?
DB_HOST=postgres_db
DB_NAME=?
TF_MODEL_ID=alaggung/bart-r3f
LLM_MODEL_PATH=/app/models/solar-10.7b-v1.5.Q5_K_M.gguf
HF_HOME=/app/models/.cache
```

- you can put any thing you want for `DB_USERNAME`, `DB_PASSWORD`, and `DB_NAME`
  - but you have to remember it to access posgres db container's shell using `psql -U {DB_USERNAME}`
- `DB_HOST` should be fixed as `postgres_db`
  - if you want to change this, you have to change the service name of `postgres_db` on `docker-compose.yml` at the same time
- `TF_MODEL_ID` can be changed if you want to tryother huggingface models
- `LLM_MODEL_PATH` should be prefixed with `/app/models/` since it is the directory path that we saved local model in `model_data` volume
  - here, `solar-10.7b-v1.5.Q5_K_M.gguf` file used as an example
  - it should point the exact `.gguf` file you want to use
- huggingface caches (`HF_HOME`) will be saved also in the volume for reusability

### 3. Build docker compose

Build docker image files from `docker-compose.yml`

```bash
docker-compose build
```

- 3 images will be newly created or downloaded
  - `iloveonsen/retrospecive-api:latest` : fast-api app image based on [``iloveonsen/fastapi-llamacpp-conda`](https://hub.docker.com/r/iloveonsen/fastapi-llamacpp-conda)
  - [`pgvector/pgvector:0.6.1-pg16`](https://hub.docker.com/r/pgvector/pgvector) : postgres image with vector store support
  - `iloveonsen/retrospective-nginx:latest` : nginx image based on [`nginx:latest`](https://hub.docker.com/_/nginx)

After the build is finished, run compose

```bash
docker-compose up
```

- if you see these 3 messages in the console, loading is complete
  - `postgres-vector-db   | LOG:  database system is ready to accept connections`
  - `retrospective-nginx  | /docker-entrypoint.sh: Configuration complete; ready for start up`
  - `retrospective-api    | INFO: Application startup complete.`
  - each log is from each container services defined in `docker-compose.yml`

### 4. Get access

Go into google chrome and type `localhost`

```http
localhost
```

If you see `{"message":"This is the root of retrospective service api."}` , installation is complete.

- since this compose use `nginx` as reverse-proxy, you don't need to put port number (8000) here
- `nginx` will heard http request through port 80 and forwards request to fastapi app running inside of `retrospective_app` container

Go into `localhost/docs` to see if api endpoint works as expected

### 5. Stop running containers

```bash
docker-compose down
```

or click pause button inside of docker desktop's `Containers` page
