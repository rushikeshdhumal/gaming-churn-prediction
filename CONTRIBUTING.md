# Contributing to Gaming Player Behavior Analysis & Churn Prediction

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the Gaming Player Behavior Analysis & Churn Prediction system.

## 🎮 Project Overview

This project analyzes gaming player behavior patterns and predicts player churn using advanced machine learning techniques. It's designed as a comprehensive data science portfolio project demonstrating end-to-end workflow from data collection to model deployment.

**Author**: Rushikesh Dhumal  
**Email**: r.dhumal@rutgers.edu  
**Repository**: https://github.com/rushikeshdhumal/gaming-churn-prediction

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)
- Basic knowledge of data science and machine learning

### Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/rushikeshdhumal/gaming-churn-prediction.git
   cd gaming-churn-prediction
   ```

2. **Set Up Development Environment**
   ```bash
   # Quick setup using Makefile
   make dev-setup
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   make install-dev
   
   # Or manual setup
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e .[dev]
   ```

3. **Initialize Project**
   ```bash
   make setup
   ```

4. **Verify Installation**
   ```bash
   make test
   make run-quick
   ```

## 🛠️ Development Workflow

### Branch Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/feature-name`: New features
- `bugfix/bug-description`: Bug fixes
- `hotfix/critical-fix`: Critical production fixes

### Making Changes

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   make qa  # Runs formatting, linting, and tests
   ```

4. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Use the provided PR template
   - Ensure all checks pass
   - Request review from maintainers

## 📝 Coding Standards

### Python Code Style

We follow PEP 8 with some modifications:

- **Line Length**: 100 characters maximum
- **Formatting**: Use `black` for automatic formatting
- **Imports**: Follow PEP 8 import order
- **Docstrings**: Use Google-style docstrings

```python
def example_function(param1: str, param2: int) -> Dict[str, Any]:
    """
    Example function demonstrating our coding style.
    
    Args:
        param1: Description of the first parameter
        param2: Description of the second parameter
        
    Returns:
        Dictionary containing the results
        
    Raises:
        ValueError: If parameters are invalid
    """
    pass
```

### Code Quality Tools

- **Formatting**: `black src/ --line-length=100`
- **Linting**: `flake8 src/ --max-line-length=100`
- **Type Checking**: `mypy src/ --ignore-missing-imports`
- **Testing**: `pytest tests/ -v --cov=src`

Use `make qa` to run all quality checks.

### File Organization

```
src/
├── data/           # Data collection and processing
├── features/       # Feature engineering
├── models/         # Model training and evaluation
├── visualization/  # Plotting and visualization
└── utils/          # Utility functions and configuration
```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── test_data/           # Test data collection and processing
├── test_features/       # Test feature engineering
├── test_models/         # Test model training and evaluation
├── test_utils/          # Test utility functions
└── fixtures/            # Test data and fixtures
```

### Writing Tests

- Use `pytest` for all tests
- Aim for >80% code coverage
- Include both unit tests and integration tests
- Use descriptive test names

```python
def test_feature_engineering_creates_expected_features():
    """Test that feature engineering creates all expected features."""
    # Arrange
    sample_data = create_sample_player_data()
    
    # Act
    feature_engineer = FeatureEngineer(sample_data)
    result = feature_engineer.create_all_features()
    
    # Assert
    expected_features = ['engagement_score', 'activity_recency', 'total_risk_score']
    for feature in expected_features:
        assert feature in result.columns
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_models/test_train_model.py -v

# Run with coverage
pytest --cov=src --cov-report=html
```

## 📊 Data Guidelines

### Data Privacy and Ethics

- Never commit real player data
- Use only synthetic or anonymized data
- Follow GDPR and privacy best practices
- Document data sources and transformations

### Data Files

- Large data files (>25MB) should not be committed
- Use `.gitignore` to exclude large datasets
- Provide instructions for obtaining external data
- Include sample data for testing

### Data Processing

- Always validate data before processing
- Handle missing values explicitly
- Document data cleaning decisions
- Maintain data lineage

## 🤖 Model Development

### Model Standards

- Use reproducible random seeds
- Document model assumptions and limitations
- Include model evaluation metrics
- Provide business interpretation of results

### Model Files

- Save models in `models/` directory
- Include model metadata (hyperparameters, performance)
- Use version control for model artifacts
- Document model deployment requirements

### Example Model Documentation

```python
# Model Metadata
MODEL_INFO = {
    "name": "xgboost_churn_predictor",
    "version": "1.0.0",
    "training_date": "2024-01-15",
    "features": 25,
    "performance": {
        "roc_auc": 0.913,
        "accuracy": 0.891,
        "precision": 0.897,
        "recall": 0.885
    },
    "hyperparameters": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1
    }
}
```

## 📋 Documentation Standards

### Code Documentation

- Document all public functions and classes
- Use type hints consistently
- Include usage examples in docstrings
- Keep documentation up to date with code changes

### Project Documentation

- Update README.md for significant changes
- Maintain accurate setup instructions
- Document configuration options
- Include troubleshooting guides

### Commit Messages

Follow conventional commit format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
- `feat(models): add XGBoost hyperparameter tuning`
- `fix(data): handle missing values in player data`
- `docs: update installation instructions`

## 🐛 Reporting Issues

### Bug Reports

Use the issue template and include:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error messages and stack traces
- Relevant logs

### Feature Requests

Include:

- Clear description of the feature
- Use case and motivation
- Proposed implementation approach
- Potential impact on existing functionality

## 🔍 Code Review Process

### Submitting Pull Requests

1. **Ensure Quality Checks Pass**
   ```bash
   make qa
   ```

2. **Update Documentation**
   - Update README if needed
   - Add/update docstrings
   - Include tests for new features

3. **Use PR Template**
   - Describe changes clearly
   - Link related issues
   - Include testing details

### Review Criteria

Reviewers will check:

- Code quality and style compliance
- Test coverage and quality
- Documentation completeness
- Performance implications
- Security considerations
- Backward compatibility

### Review Process

1. Automated checks must pass
2. At least one maintainer approval required
3. Address all review feedback
4. Squash commits before merge (if requested)

## 🚀 Release Process

### Version Numbers

We use Semantic Versioning (SemVer):
- MAJOR: Incompatible API changes
- MINOR: Backward-compatible functionality
- PATCH: Backward-compatible bug fixes

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version number bumped
- [ ] CHANGELOG.md updated
- [ ] Performance benchmarks run
- [ ] Security scan completed

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn and contribute
- Acknowledge others' contributions

### Communication

- Use GitHub issues for bugs and features
- Be clear and concise in communications
- Provide context and examples
- Ask questions if unsure

## 📚 Learning Resources

### Data Science & Machine Learning

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

### Gaming Analytics

- [Steam Web API Documentation](https://steamcommunity.com/dev)
- [Game Analytics Best Practices](https://gameanalytics.com/docs/)

### Software Development

- [Python PEP 8 Style Guide](https://pep8.org/)
- [Git Best Practices](https://git-scm.com/book)
- [Docker Documentation](https://docs.docker.com/)

## 🆘 Getting Help

### Documentation

1. Check the README.md
2. Review existing issues
3. Read the code documentation
4. Check the wiki (if available)

### Asking Questions

1. Search existing issues first
2. Provide minimal reproducible example
3. Include environment details
4. Be specific about the problem

### Contacting Maintainers

- **GitHub Issues**: For bugs and feature requests
- **Email**: r.dhumal@rutgers.edu (for sensitive issues)
- **Discussions**: Use GitHub Discussions for general questions

## 🙏 Acknowledgments

Thank you for contributing to this project! Your contributions help make this a better resource for the data science and gaming analytics community.

### Contributors

- Rushikesh Dhumal - Project Creator and Maintainer

### Special Thanks

- Steam for providing the Web API
- Kaggle community for curated datasets
- Open source libraries that make this project possible

---

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the same MIT License that covers the project. See [LICENSE](LICENSE) file for details.

---

*This contributing guide is a living document. Please suggest improvements to help make it better for everyone!*