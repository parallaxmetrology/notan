# Contributing

Most useful contributions come from people who look at pictures for a living.

- **Readings that look wrong.** A painting where the mass map or the 3-value Notan disagrees with what a painter sees is the most useful report there is. Say what you see and what the tool says.
- **Construction readings.** Pictures you know to be built from colour (or from value) and what `dependence_map` shows for them. Equal-luminance passages, in the sense of Margaret Livingstone's *Vision and Art*, are especially welcome.
- **New known-answer images.** A synthetic image whose right answer is not in doubt, and which the current measures get wrong, belongs in `stimuli.py`.

## Development

```bash
git clone https://github.com/parallaxmetrology/notan
cd notan
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest
```

`tests/test_gates.py` holds predictions that were written before the code was trusted. Change a gate only with a reason in the changelog.

Be kind and specific. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
