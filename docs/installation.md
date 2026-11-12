# Installation

```bash
pip install scalecov             # after the first release
pip install "scalecov[plot]"     # with matplotlib for plots
```

From source:

```bash
git clone https://github.com/vubcfe/scalecov && cd scalecov
pip install -e ".[dev]"
pytest
```

Requirements: Python 3.9 or later, numpy, pandas 2.2 or later and scipy. Linux, macOS and Windows are supported. No compiled code is involved.
