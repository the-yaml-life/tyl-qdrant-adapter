# Pull Request

## 📋 Description

<!-- Provide a brief description of the changes in this PR -->

## 🔗 Related Issue

<!-- Link to the issue this PR addresses -->
Closes #

## 🛠️ Type of Change

<!-- Mark the relevant option with an [x] -->

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🔧 Maintenance (dependency updates, CI, etc.)
- [ ] 🔄 Schema migration related

## 🧪 Testing

<!-- Describe the tests you ran and how to reproduce them -->

- [ ] Unit tests pass (`cargo test --lib`)
- [ ] Integration tests pass (`cargo test --test integration_tests`)
- [ ] Migration tests pass (`cargo test --features schema-migration --test migration_tests`)
- [ ] Examples work (`cargo run --example basic_usage`)
- [ ] All features compile (`cargo check --all-features`)

## 📝 Checklist

<!-- Mark completed items with [x] -->

- [ ] My code follows TYL framework patterns and conventions
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings (`cargo clippy`)
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have checked that my changes work with both `QdrantAdapter` and `MockQdrantAdapter`

## 🔄 Migration Changes (if applicable)

<!-- Fill this section if your PR includes schema migration changes -->

- [ ] Migration is reversible
- [ ] Migration has proper dependency tracking
- [ ] Pact contracts are updated
- [ ] Breaking changes are documented

## 📚 Documentation

<!-- List any documentation that was added or updated -->

- [ ] README.md updated
- [ ] CLAUDE.md updated
- [ ] Code comments added
- [ ] Examples updated

## 💔 Breaking Changes

<!-- List any breaking changes and migration guide -->

- None

## 📸 Screenshots (if applicable)

<!-- Add screenshots for UI changes or console output -->

## 🎯 Additional Context

<!-- Add any other context about the PR here -->