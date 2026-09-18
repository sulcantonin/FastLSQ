#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Patch a freshly minted Zenodo concept DOI into every place it belongs.

Zenodo mints two DOIs for a GitHub-archived repository:

  * a **concept DOI**, which always resolves to the newest archived version, and
  * a **version DOI**, one per release.

Cite the concept DOI.  It is the one that keeps working when you publish 0.7.0.

This script writes it into the four files that have to agree with each other:

    README.md        the badge and the citation section
    CITATION.cff     the identifiers block, for GitHub's "Cite this repository"
    site/index.html  the footer and the cite section
    .zenodo.json     as a related identifier, so the deposit points at itself

Run:
    python scripts/add_zenodo_doi.py 10.5281/zenodo.1234567
    python scripts/add_zenodo_doi.py 10.5281/zenodo.1234567 --check
"""

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

DOI_RE = re.compile(r"^10\.5281/zenodo\.(\d+)$")


def fail(msg):
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def patch_readme(doi, record, dry):
    p = ROOT / "README.md"
    s = original = p.read_text()

    badge = (f"[![DOI](https://zenodo.org/badge/DOI/{doi}.svg)]"
             f"(https://doi.org/{doi})")
    if "zenodo.org/badge" in s:
        s = re.sub(r"\[!\[DOI\]\([^)]*\)\]\([^)]*\)", badge, s, count=1)
    else:
        # place it last in the existing badge run, which ends at the arXiv badge
        anchor = "[![arXiv](https://img.shields.io/badge/arXiv-2602.10541-b31b1b.svg)](https://arxiv.org/abs/2602.10541)"
        if anchor not in s:
            fail("README.md: could not find the arXiv badge to anchor the DOI badge to")
        s = s.replace(anchor, anchor + "\n" + badge, 1)

    # A software-citation note beside the paper citation.  Guard on the doi.org
    # link the note itself contains -- an earlier version guarded on "zenodo.org",
    # which only ever appears in the badge, so a second run appended the note twice.
    note = (f"To cite a specific archived version of the code rather than the paper, "
            f"use the Zenodo concept DOI [{doi}](https://doi.org/{doi}), which always "
            f"resolves to the most recent release.\n")
    marker = "## License"
    tail = s.split("## Citing this work")[-1].split(marker)[0]
    if "doi.org/10.5281/zenodo" not in tail:
        s = s.replace(marker, note + "\n" + marker, 1)
    else:
        s = re.sub(r"To cite a specific archived version of the code[^\n]*\n", note, s, count=1)

    return write(p, original, s, dry)


def patch_citation(doi, record, dry):
    p = ROOT / "CITATION.cff"
    s = original = p.read_text()

    block = (f"identifiers:\n"
             f"  - type: doi\n"
             f"    value: {doi}\n"
             f"    description: Concept DOI — resolves to the latest archived version\n")

    if re.search(r"^identifiers:", s, flags=re.M):
        s = re.sub(r"^identifiers:\n(?:  .*\n)+", block, s, count=1, flags=re.M)
    else:
        # replace the commented-out placeholder the repo ships with
        commented = re.search(
            r"# Once the Zenodo DOI is minted.*?#     description: Concept DOI[^\n]*\n",
            s, flags=re.S)
        if commented:
            s = s[:commented.start()] + block + s[commented.end():]
        else:
            s = s.replace("preferred-citation:", block + "\npreferred-citation:", 1)

    return write(p, original, s, dry)


def patch_site(doi, record, dry):
    p = ROOT / "site" / "index.html"
    s = original = p.read_text()

    link = f'<a href="https://doi.org/{doi}">software DOI</a>'
    if "doi.org/10.5281/zenodo" in s:
        s = re.sub(r'<a href="https://doi\.org/10\.5281/zenodo\.[^"]*">[^<]*</a>',
                   link, s)
    else:
        anchor = '<a href="https://arxiv.org/abs/2602.10541">paper (arXiv)</a>'
        if anchor not in s:
            fail("site/index.html: could not find the arXiv footer link to anchor to")
        s = s.replace(anchor, anchor + "\n      " + link, 1)

    # and in the BibTeX block, so a copied citation carries the DOI
    if "zenodo" not in s.split("<code id=\"bib\">")[-1].split("</code>")[0]:
        s = s.replace(
            '  url           = {https://arxiv.org/abs/2602.10541}\n}',
            '  url           = {https://arxiv.org/abs/2602.10541}\n}\n\n'
            '@software{sulc2026fastlsq_software,\n'
            '  author    = {Sulc, Antonin},\n'
            '  title     = {{FastLSQ}},\n'
            f'  doi       = {{{doi}}},\n'
            f'  url       = {{https://doi.org/{doi}}},\n'
            '  publisher = {Zenodo}\n}', 1)

    return write(p, original, s, dry)


def patch_zenodo_json(doi, record, dry):
    p = ROOT / ".zenodo.json"
    d = json.loads(p.read_text())
    original = p.read_text()

    rel = [r for r in d.get("related_identifiers", [])
           if "zenodo" not in r.get("identifier", "")]
    rel.append({
        "identifier": f"https://doi.org/{doi}",
        "relation": "isVersionOf",
        "resource_type": "software",
        "scheme": "doi",
    })
    d["related_identifiers"] = rel
    return write(p, original, json.dumps(d, indent=2, ensure_ascii=False) + "\n", dry)


def write(p, before, after, dry):
    rel = p.relative_to(ROOT)
    if before == after:
        print(f"  {rel}: already up to date")
        return False
    if dry:
        print(f"  {rel}: WOULD change")
        return True
    p.write_text(after)
    print(f"  {rel}: updated")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("doi", help="Zenodo CONCEPT doi, e.g. 10.5281/zenodo.1234567")
    ap.add_argument("--check", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args()

    doi = args.doi.strip().removeprefix("https://doi.org/").removeprefix("doi:")
    m = DOI_RE.match(doi)
    if not m:
        fail(f"{args.doi!r} is not a Zenodo DOI (expected 10.5281/zenodo.NNNNNNN)")

    print(f"{'Checking' if args.check else 'Writing'} DOI {doi}")
    changed = [fn(doi, m.group(1), args.check) for fn in
               (patch_readme, patch_citation, patch_site, patch_zenodo_json)]

    if args.check:
        print(f"\n{sum(changed)} file(s) would change.")
    elif any(changed):
        print(f"\n{sum(changed)} file(s) updated. Review with `git diff`, then commit.")
        print("Note: the site redeploys automatically once site/ lands on main.")
    else:
        print("\nNothing to do.")


if __name__ == "__main__":
    main()
