.PHONY: api ui

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

api: ## Run the API server
	poetry run uvicorn src.server.server:app --reload

ui: ## Run the UI server
	cd frontend && poetry run python -m http.server 8080
