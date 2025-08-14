# Makefile for vLLM OpenAI API with Mistral-7B

# Configuration
PORT ?= 8000
MODEL ?= mistralai/Mistral-7B-Instruct-v0.2
API_KEY ?= 21ce55ad0efe29f09492e22c0b34d4c361f070f1b96531379403810f911bdf89
DOCKER_IMAGE ?= vllm/vllm-openai

.PHONY: run clean

run:
	@echo "Starting vLLM OpenAI API server with $(MODEL)..."
	docker run --gpus all -it --rm -p $(PORT):$(PORT) \
		$(DOCKER_IMAGE) \
		--model $(MODEL) \
		--api-key $(API_KEY)

clean:
	@echo "Cleaning up Docker containers..."
	-docker stop $$(docker ps -q --filter ancestor=$(DOCKER_IMAGE))


# docker compose exec ollama ollama pull mistral
# docker compose exec ollama ollama pull bge-small-en-v1.5
