# Gaming Player Behavior Analysis & Churn Prediction
# Makefile for project automation and workflow management
#
# Author: Rushikesh Dhumal
# Email: r.dhumal@rutgers.edu

.PHONY: help setup install install-dev clean test lint format run-analysis \
        run-quick setup-db collect-data train-models predict deploy \
        docker-build docker-run validate-config create-dirs \
        generate-reports backup restore

# Default target
.DEFAULT_GOAL := help

# Project configuration
PROJECT_NAME := gaming-churn-prediction
PYTHON := python3
PIP := pip3
VENV := venv
VENV_ACTIVATE := $(VENV)/bin/activate
ENVIRONMENT := development

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)Gaming Player Behavior Analysis & Churn Prediction$(NC)"
	@echo "$(GREEN)====================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "  make setup          # Complete project setup"
	@echo "  make run-quick      # Run quick analysis"
	@echo "  make run-analysis   # Run complete analysis"

setup: ## Complete project setup (recommended for first time)
	@echo "$(GREEN)🚀 Setting up Gaming Churn Prediction project...$(NC)"
	$(MAKE) create-dirs
	$(MAKE) install-dev
	$(MAKE) setup-db
	$(MAKE) validate-config
	@echo "$(GREEN)✅ Project setup completed successfully!$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  make run-quick      # Run quick analysis (5-10 minutes)"
	@echo "  make run-analysis   # Run complete analysis (15-30 minutes)"

create-dirs: ## Create necessary project directories
	@echo "$(GREEN)📁 Creating project directories...$(NC)"
	@mkdir -p data/{raw,processed,external}
	@mkdir -p models reports/figures logs scripts/outputs tests docs
	@touch data/processed/.gitkeep data/external/.gitkeep
	@touch models/.gitkeep reports/figures/.gitkeep
	@touch logs/.gitkeep scripts/outputs/.gitkeep
	@touch tests/.gitkeep docs/.gitkeep
	@echo "$(GREEN)✅ Directories created$(NC)"

install: ## Install project dependencies
	@echo "$(GREEN)📦 Installing dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

install-dev: ## Install development dependencies
	@echo "$(GREEN)📦 Installing development dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install -e .[dev]
	@echo "$(GREEN)✅ Development environment ready$(NC)"

setup-venv: ## Create and setup virtual environment
	@echo "$(GREEN)🐍 Setting up virtual environment...$(NC)"
	$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)✅ Virtual environment created$(NC)"
	@echo "$(YELLOW)Activate with: source $(VENV_ACTIVATE)$(NC)"

clean: ## Clean up generated files and caches
	@echo "$(GREEN)🧹 Cleaning up...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/
	rm -rf .pytest_cache/ .mypy_cache/
	rm -f logs/*.log
	rm -f scripts/outputs/*
	@echo "$(GREEN)✅ Cleanup completed$(NC)"

test: ## Run test suite
	@echo "$(GREEN)🧪 Running tests...$(NC)"
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)✅ Tests completed$(NC)"

lint: ## Run code linting
	@echo "$(GREEN)🔍 Running linters...$(NC)"
	flake8 src/ --max-line-length=100 --ignore=E203,W503
	mypy src/ --ignore-missing-imports
	@echo "$(GREEN)✅ Linting completed$(NC)"

format: ## Format code with black
	@echo "$(GREEN)🎨 Formatting code...$(NC)"
	black src/ scripts/ database/ --line-length=100
	@echo "$(GREEN)✅ Code formatted$(NC)"

validate-config: ## Validate project configuration
	@echo "$(GREEN)⚙️ Validating configuration...$(NC)"
	$(PYTHON) -c "from src.utils.config import ConfigManager; cm = ConfigManager('$(ENVIRONMENT)'); result = cm.validate_config(); print('✅ Config valid' if result['valid'] else '❌ Config invalid'); [print(f'  Error: {e}') for e in result['errors']]; [print(f'  Warning: {w}') for w in result['warnings']]"

setup-db: ## Initialize database
	@echo "$(GREEN)🗄️ Setting up database...$(NC)"
	gaming-churn-setup-db || $(PYTHON) database/setup_database.py
	@echo "$(GREEN)✅ Database initialized$(NC)"

collect-data: ## Collect and generate data
	@echo "$(GREEN)📊 Collecting data...$(NC)"
	gaming-churn-collect-data || $(PYTHON) src/data/data_collector.py
	@echo "$(GREEN)✅ Data collection completed$(NC)"

train-models: ## Train machine learning models
	@echo "$(GREEN)🤖 Training models...$(NC)"
	gaming-churn-train || $(PYTHON) src/models/train_model.py
	@echo "$(GREEN)✅ Model training completed$(NC)"

predict: ## Make predictions with trained models
	@echo "$(GREEN)🔮 Making predictions...$(NC)"
	gaming-churn-predict || $(PYTHON) src/utils/deployment_utils.py
	@echo "$(GREEN)✅ Predictions completed$(NC)"

run-quick: ## Run quick analysis (small dataset, essential models)
	@echo "$(GREEN)🏃‍♂️ Running quick analysis...$(NC)"
	@echo "$(YELLOW)This will take 5-10 minutes$(NC)"
	$(PYTHON) scripts/run_complete_analysis.py --environment $(ENVIRONMENT) --quick-run
	@echo "$(GREEN)✅ Quick analysis completed$(NC)"

run-analysis: ## Run complete analysis pipeline
	@echo "$(GREEN)🔬 Running complete analysis...$(NC)"
	@echo "$(YELLOW)This will take 15-30 minutes$(NC)"
	$(PYTHON) scripts/run_complete_analysis.py --environment $(ENVIRONMENT)
	@echo "$(GREEN)✅ Complete analysis finished$(NC)"

run-production: ## Run analysis in production mode
	@echo "$(GREEN)🏭 Running production analysis...$(NC)"
	$(PYTHON) scripts/run_complete_analysis.py --environment production
	@echo "$(GREEN)✅ Production analysis completed$(NC)"

generate-reports: ## Generate analysis reports
	@echo "$(GREEN)📋 Generating reports...$(NC)"
	$(PYTHON) -c "from scripts.run_complete_analysis import ComprehensiveAnalysisPipeline; p = ComprehensiveAnalysisPipeline('$(ENVIRONMENT)'); p.step9_generate_reports()"
	@echo "$(GREEN)✅ Reports generated in scripts/outputs/$(NC)"

deploy: ## Deploy models for production use
	@echo "$(GREEN)🚀 Deploying models...$(NC)"
	$(PYTHON) -c "from src.utils.deployment_utils import deploy_model_to_production; result = deploy_model_to_production(); print('✅ Deployment successful' if result['status'] == 'SUCCESS' else '❌ Deployment failed')"
	@echo "$(GREEN)✅ Models deployed$(NC)"

# Docker commands
docker-build: ## Build Docker image
	@echo "$(GREEN)🐳 Building Docker image...$(NC)"
	docker build -t $(PROJECT_NAME):latest .
	@echo "$(GREEN)✅ Docker image built$(NC)"

docker-run: ## Run project in Docker container
	@echo "$(GREEN)🐳 Running in Docker...$(NC)"
	docker run -it --rm -v $(PWD)/data:/app/data $(PROJECT_NAME):latest
	@echo "$(GREEN)✅ Docker run completed$(NC)"

# Data management
backup: ## Backup important project data
	@echo "$(GREEN)💾 Creating backup...$(NC)"
	@mkdir -p backups
	tar -czf backups/gaming-churn-backup-$(shell date +%Y%m%d_%H%M%S).tar.gz \
		data/ models/ reports/ logs/ scripts/outputs/ \
		--exclude='data/external/*' --exclude='*.log' 2>/dev/null || true
	@echo "$(GREEN)✅ Backup created in backups/$(NC)"

restore: ## Restore from backup (specify BACKUP_FILE=filename)
	@echo "$(GREEN)📂 Restoring from backup...$(NC)"
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "$(RED)❌ Please specify BACKUP_FILE=filename$(NC)"; \
		exit 1; \
	fi
	tar -xzf $(BACKUP_FILE)
	@echo "$(GREEN)✅ Restore completed$(NC)"

# Development tools
dev-setup: ## Complete development environment setup
	@echo "$(GREEN)👨‍💻 Setting up development environment...$(NC)"
	$(MAKE) setup-venv
	@echo "$(YELLOW)Activate virtual environment: source $(VENV_ACTIVATE)$(NC)"
	@echo "$(YELLOW)Then run: make install-dev$(NC)"

check-deps: ## Check for outdated dependencies
	@echo "$(GREEN)🔍 Checking dependencies...$(NC)"
	$(PIP) list --outdated

update-deps: ## Update dependencies to latest versions
	@echo "$(GREEN)⬆️ Updating dependencies...$(NC)"
	$(PIP) install --upgrade -r requirements.txt

# Quality assurance
qa: ## Run full quality assurance (lint, test, format)
	@echo "$(GREEN)🏆 Running quality assurance...$(NC)"
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test
	@echo "$(GREEN)✅ Quality assurance completed$(NC)"

# Performance testing
benchmark: ## Run performance benchmarks
	@echo "$(GREEN)⏱️ Running benchmarks...$(NC)"
	$(PYTHON) -c "import time; from scripts.run_complete_analysis import ComprehensiveAnalysisPipeline; start=time.time(); p=ComprehensiveAnalysisPipeline('development', quick_run=True); p.run_complete_pipeline(); print(f'Benchmark completed in {time.time()-start:.2f} seconds')"

# Documentation
docs: ## Generate documentation
	@echo "$(GREEN)📚 Generating documentation...$(NC)"
	@echo "$(YELLOW)Documentation available in:$(NC)"
	@echo "  README.md - Main project documentation"
	@echo "  data/DATA_INFO.md - Dataset documentation"
	@echo "  reports/ - Analysis reports"
	@echo "  scripts/outputs/ - Generated reports"

# Monitoring and status
status: ## Show project status
	@echo "$(GREEN)📊 Project Status$(NC)"
	@echo "$(GREEN)===============$(NC)"
	@echo "Environment: $(ENVIRONMENT)"
	@echo "Python: $(shell $(PYTHON) --version 2>/dev/null || echo 'Not found')"
	@echo "Pip: $(shell $(PIP) --version 2>/dev/null || echo 'Not found')"
	@echo "Git: $(shell git --version 2>/dev/null || echo 'Not found')"
	@echo ""
	@echo "$(YELLOW)Project Structure:$(NC)"
	@echo "Data files: $(shell find data -name '*.csv' 2>/dev/null | wc -l)"
	@echo "Models: $(shell find models -name '*.pkl' 2>/dev/null | wc -l)"
	@echo "Reports: $(shell find reports -type f 2>/dev/null | wc -l)"
	@echo "Logs: $(shell find logs -name '*.log' 2>/dev/null | wc -l)"

# Environment management
env-dev: ## Switch to development environment
	$(eval ENVIRONMENT := development)
	@echo "$(GREEN)🔧 Switched to development environment$(NC)"

env-prod: ## Switch to production environment
	$(eval ENVIRONMENT := production)
	@echo "$(GREEN)🏭 Switched to production environment$(NC)"

env-test: ## Switch to testing environment
	$(eval ENVIRONMENT := testing)
	@echo "$(GREEN)🧪 Switched to testing environment$(NC)"

# Shortcuts for common workflows
quick: run-quick ## Alias for run-quick

full: run-analysis ## Alias for run-analysis

all: setup run-analysis generate-reports ## Complete workflow from setup to reports

# Emergency commands
emergency-reset: ## Emergency reset (clean everything and restart)
	@echo "$(RED)⚠️ EMERGENCY RESET - This will delete all generated data!$(NC)"
	@read -p "Are you sure? Type 'yes' to continue: " confirm && [ "$$confirm" = "yes" ]
	$(MAKE) clean
	rm -rf data/processed/* data/external/* models/* reports/figures/* logs/* scripts/outputs/*
	@echo "$(GREEN)🔄 Emergency reset completed$(NC)"
	@echo "$(YELLOW)Run 'make setup' to reinitialize$(NC)"

# Help for specific topics
help-analysis: ## Show analysis workflow help
	@echo "$(GREEN)🔬 Analysis Workflow Help$(NC)"
	@echo "$(GREEN)========================$(NC)"
	@echo ""
	@echo "$(YELLOW)Quick Start (5-10 minutes):$(NC)"
	@echo "  make setup          # One-time setup"
	@echo "  make run-quick      # Quick analysis"
	@echo ""
	@echo "$(YELLOW)Complete Analysis (15-30 minutes):$(NC)"
	@echo "  make setup          # One-time setup"
	@echo "  make run-analysis   # Full analysis"
	@echo ""
	@echo "$(YELLOW)Step-by-step:$(NC)"
	@echo "  make setup-db       # Initialize database"
	@echo "  make collect-data   # Collect data"
	@echo "  make train-models   # Train ML models"
	@echo "  make predict        # Make predictions"
	@echo "  make generate-reports # Create reports"

help-development: ## Show development workflow help
	@echo "$(GREEN)👨‍💻 Development Workflow Help$(NC)"
	@echo "$(GREEN)=============================$(NC)"
	@echo ""
	@echo "$(YELLOW)First-time setup:$(NC)"
	@echo "  make dev-setup      # Create virtual environment"
	@echo "  source venv/bin/activate"
	@echo "  make install-dev    # Install dependencies"
	@echo ""
	@echo "$(YELLOW)Daily workflow:$(NC)"
	@echo "  make format         # Format code"
	@echo "  make lint          # Check code quality"
	@echo "  make test          # Run tests"
	@echo "  make qa            # All quality checks"

# Version information
version: ## Show version information
	@echo "$(GREEN)Gaming Player Behavior Analysis & Churn Prediction$(NC)"
	@echo "$(GREEN)Version: 1.0.0$(NC)"
	@echo "$(GREEN)Author: Rushikesh Dhumal$(NC)"
	@echo "$(GREEN)Email: r.dhumal@rutgers.edu$(NC)"