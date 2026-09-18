# Releasing FastLSQ

## Cutting a release

1. Bump the version in `pyproject.toml`. `tests/test_vector_basis.py::test_version`
   compares it against `fastlsq.__version__`, so the suite fails if the two drift.
2. Add the `CHANGELOG.md` entry. Say what was a *silent wrong answer* separately
   from what merely raised an error — the two matter very differently to a user.
3. Merge to `main`, then tag the merge commit:

   ```bash
   git tag -a v0.7.0 -m "FastLSQ 0.7.0" && git push origin v0.7.0
   ```

   **Tag the commit you actually build from.** Three early releases (0.1.2, 0.1.3,
   0.1.5) were cut from working trees holding uncommitted source, and their
   published sdists match no commit in this repository. `apebench.py` shipped to
   PyPI twice and exists nowhere in git. See *Release tags* in `CHANGELOG.md`.

4. Build and check:

   ```bash
   python -m build
   twine check dist/*
   ```

5. Create the GitHub Release from the tag, with the `CHANGELOG.md` entry as its
   body. **Do this before uploading to PyPI** if Zenodo archiving is on — see below.
6. `twine upload dist/*`.

## Zenodo archiving

Zenodo mints a DOI for each GitHub Release, so the code is citable.

JOSS requires this, but *at acceptance, not at submission*: after review completes
you make a tagged release, deposit it with Zenodo or figshare, and post the version
and DOI to the review thread. So there is no rush to have a DOI before submitting —
only before the paper is accepted.

### One-time setup

1. Sign in at [zenodo.org](https://zenodo.org) **with GitHub** and authorise the
   integration.
2. Under *GitHub* in your Zenodo account, flip the switch on for
   `sulcantonin/FastLSQ`.
3. Create a GitHub Release.

> **The gotcha:** Zenodo only archives releases created *after* the switch is
> enabled. The 20 existing releases will not be archived retroactively. To get a
> DOI without inventing a version, either cut the next real release, or delete and
> recreate the most recent one *after* enabling — the tag can stay where it is.

`.zenodo.json` in the repository root controls how the deposit is described:
title, abstract, author with ORCID, affiliation, licence, keywords, and the link
back to the arXiv preprint. Without it Zenodo guesses from the repository, and
guesses poorly. Keep it in step with `CITATION.cff`.

### After the first DOI is minted

Zenodo gives two DOIs:

| | what it resolves to | use it for |
|---|---|---|
| **Concept DOI** | always the newest archived version | the README badge, `CITATION.cff`, anything citing "FastLSQ" generally |
| **Version DOI** | one specific release, forever | reproducibility — a paper pinning the exact code it ran |

Cite the **concept DOI**. Write it into the four files that must agree:

```bash
python scripts/add_zenodo_doi.py 10.5281/zenodo.XXXXXXX --check   # preview
python scripts/add_zenodo_doi.py 10.5281/zenodo.XXXXXXX           # write
```

It patches `README.md`, `CITATION.cff` and `.zenodo.json`, and is idempotent — rerunning with a new DOI replaces rather than appends.
