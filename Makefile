.PHONY: build train index-rag index-rag-cnn index-mode index-all fit-pca fit-autoencoder fit-all api evaluate evaluate-all test clean

build:
	docker compose build

train:
	docker compose run --rm trainer

index-rag:
	EMBEDDING_MODE=raw784 docker compose run --rm rag-indexer

index-rag-cnn:
	EMBEDDING_MODE=cnn_embedding docker compose run --rm rag-indexer

index-mode:
	EMBEDDING_MODE=$(MODE) docker compose run --rm rag-indexer

index-all:
	@for mode in raw784 blur784 hog hog_blur cnn_embedding; do \
		echo "=== Indexing $$mode ==="; \
		EMBEDDING_MODE=$$mode docker compose run --rm rag-indexer; \
	done
	@for mode in pca64 pca128 pca256 autoencoder64 autoencoder128 autoencoder256; do \
		echo "=== Indexing $$mode ==="; \
		EMBEDDING_MODE=$$mode docker compose run --rm rag-indexer; \
	done

fit-pca:
	@for dim in 64 128 256; do \
		echo "=== Fitting pca$$dim ==="; \
		docker compose run --rm fitter python -m fashion_compare.rag.fit --mode pca$$dim; \
	done

fit-autoencoder:
	@for dim in 64 128 256; do \
		echo "=== Fitting autoencoder$$dim ==="; \
		docker compose run --rm fitter python -m fashion_compare.rag.fit --mode autoencoder$$dim; \
	done

fit-all: fit-pca fit-autoencoder

api:
	docker compose up -d classifier-api rag-api

evaluate:
	docker compose run --rm evaluator

evaluate-all:
	docker compose run --rm evaluator python -m fashion_compare.evaluation.compare --tune-top-k

test:
	docker compose run --rm trainer pytest -q

clean:
	docker compose down
	rm -rf artifacts/classifier/* artifacts/rag/* reports/*
