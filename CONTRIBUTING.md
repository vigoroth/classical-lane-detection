# Contributing to Classical Lane Detection

Thank you for your interest in contributing to this project! This is a learning project from a 12-project thesis path on 3D lane detection. Contributions are welcome and appreciated.

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker to report bugs
- Describe the issue clearly with steps to reproduce
- Include your Python version and operating system
- Attach relevant error messages or screenshots

### Suggesting Enhancements

- Open an issue to discuss your proposed enhancement
- Explain the use case and benefits
- Be open to feedback and alternative approaches

### Submitting Pull Requests

1. **Fork the repository**
   ```bash
   # Click the "Fork" button on GitHub, then clone your fork
   git clone https://github.com/YOUR-USERNAME/classical-lane-detection.git
   cd classical-lane-detection
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-improvement
   ```

3. **Make your changes**
   - Write clear, documented code
   - Follow existing code style and conventions
   - Add comments where logic isn't self-evident

4. **Add tests**
   - All new features should include tests
   - Ensure existing tests still pass
   - Aim to maintain or improve code coverage (currently 63%)

5. **Run the test suite**
   ```bash
   # Run all tests
   pytest tests/ -v

   # Check coverage
   pytest tests/ --cov=src --cov-report=term

   # Ensure all tests pass before committing
   ```

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: brief description"

   # Use descriptive commit messages
   # Follow conventional commits format if possible
   ```

7. **Push to your fork**
   ```bash
   git push origin feature/your-improvement
   ```

8. **Create a Pull Request**
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Select your fork and branch
   - Describe your changes clearly
   - Link any related issues

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Keep functions focused on a single responsibility
- Add docstrings to functions and classes

### Testing

- Write unit tests for new functions
- Write integration tests for new features
- Test edge cases and error conditions
- Mock external dependencies where appropriate

### Documentation

- Update relevant documentation for your changes
- Add docstrings with parameter descriptions
- Update README.md if adding new features
- Include code examples where helpful

## Project Structure

```
project-01-classical-lanes/
├── src/              # Source code
│   ├── preprocessing.py
│   ├── edge_detection.py
│   ├── line_detection.py
│   ├── main.py
│   └── analysis/     # Analysis tools
├── tests/            # Test suite
├── docs/             # Documentation
├── data/             # Test data
└── results/          # Output results
```

## Testing Checklist

Before submitting a pull request, ensure:

- [ ] All tests pass locally
- [ ] New code has test coverage
- [ ] Code follows project style guidelines
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive
- [ ] No unnecessary files are included

## Questions?

If you have questions about contributing, feel free to:
- Open an issue for discussion
- Review existing issues and pull requests
- Check the [documentation](docs/README.md)

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers and beginners
- Focus on what's best for the project
- Show empathy towards other contributors

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Classical Lane Detection! 🚗
