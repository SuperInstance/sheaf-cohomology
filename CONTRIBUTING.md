# Contributing — sheaf-cohomology

> *Thank you for contributing to the SuperInstance fleet.*

## How to Contribute

### 1. Understand the Ternary Ethos

We use {-1, 0, +1} because it is the minimum viable alphabet for expressing agreement,
disagreement, and abstention. **0 is not "nothing"** — it is a deliberate neutral state.

### 2. Set Up

```bash
git clone https://github.com/SuperInstance/sheaf-cohomology.git
cd sheaf-cohomology
```

### 3. Code Standards

- **No unsafe code** unless absolutely necessary
- **Ternary-compatible** — prefer {-1, 0, +1} where appropriate
- **Test coverage** — every public function needs tests
- **Documentation** — all public items must have doc comments

### 4. Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add ternary quantization
fix: handle zero-input edge case
docs: update architecture
```

### 5. Pull Request Process

1. Fork and feature branch
2. Write tests
3. All tests pass: `cargo test`
4. Submit PR with clear description

## License

By contributing, you agree your contributions will be MIT OR Apache-2.0.
