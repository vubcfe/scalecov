# Contributing

Bug reports, questions and pull requests are welcome on the [issue tracker](https://github.com/vubcfe/scalecov/issues).

## Development setup

```bash
git clone https://github.com/vubcfe/scalecov && cd scalecov
pip install -e ".[dev]"
pytest                          # tests (offline, a few seconds)
ruff check src tests examples   # style
mkdocs serve                    # documentation preview
```

## Guidelines

- Every new feature comes with a test. Prefer property tests on `scalecov.simulate`, where the correct answer is known (for example, a construction that must reach nominal coverage).
- Keep the package free of heavy dependencies: numpy, pandas and scipy only; plotting stays optional.
- Document user-visible changes in `CHANGELOG.md`.
- Start every source file with the SPDX licence header used in the existing files.
- Be respectful; we follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
