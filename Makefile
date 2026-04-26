.PHONY: build train index-rag index-rag-cnn api evaluate test clean

build:
	docker compose build

train:
	docker compose run --rm trainer

index-rag:
	EMBEDDING_MODE=raw784 docker compose run --rm rag-indexer

index-rag-cnn:
	EMBEDDING_MODE=cnn_embedding docker compose run --rm rag-indexer

api:
	docker compose up -d classifier-api rag-api

evaluate:
	docker compose run --rm evaluator

test:
	docker compose run --rm trainer pytest -q

clean:
	docker compose down
	rm -rf artifacts/classifier/* artifacts/rag/* reports/*

