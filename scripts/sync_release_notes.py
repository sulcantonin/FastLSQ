#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Rewrite GitHub Release bodies so they survive Zenodo's Markdown rendering.

Zenodo takes a GitHub Release's body and converts it to the HTML of the archived
record's description.  That conversion does **not** understand Markdown tables: a
table arrives as one flattened run of pipes, like

    | Feature | API | Benefit | |---|---|---| | Multi-axis integrals | ...

which is unreadable, and is the first thing anyone following the DOI sees.

Other Markdown — headings, lists, code fences, emphasis — converts fine.  So the
fix is narrow: turn each table into a definition-style list, and leave everything
else alone.  `CHANGELOG.md` keeps its tables, because GitHub renders those
correctly and they are easier to read there.

    | Feature | API | Benefit |          ### Feature — `API`
    |---|---|---|                 ->     Benefit
    | Multi-axis | `Op.x` | Fast |

Run:
    python scripts/sync_release_notes.py --check        # show what would change
    python scripts/sync_release_notes.py                # rewrite every release
    python scripts/sync_release_notes.py v0.6.0         # just one
"""

import argparse
import json
import re
import subprocess
import sys


def gh(*args, **kw):
    return subprocess.run(["gh", *args], capture_output=True, text=True, **kw)


def split_row(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def is_sep(line):
    return bool(re.fullmatch(r"\s*\|?[\s:|-]+\|?\s*", line)) and "-" in line


def flatten_tables(md):
    """Replace every Markdown table with a heading-and-prose block."""
    out, i = [], 0
    lines = md.replace("\r\n", "\n").split("\n")
    while i < len(lines):
        # a table is a header row, a separator row, then one or more body rows
        if (lines[i].lstrip().startswith("|")
                and i + 1 < len(lines) and is_sep(lines[i + 1])):
            header = split_row(lines[i])
            i += 2
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            out.append("")
            for r in rows:
                cells = (r + [""] * len(header))[:len(header)]
                # first cell is the subject; second, when it looks like an API
                # name, joins it on the same line; the rest become labelled prose
                title = cells[0]
                rest = cells[1:]
                if rest and len(rest[0]) < 80 and "`" in rest[0]:
                    out.append(f"**{title}** — {rest[0]}")
                    rest = rest[1:]
                else:
                    out.append(f"**{title}**")
                for label, val in zip(header[len(cells) - len(rest):], rest):
                    if val:
                        out.append(f"{label}: {val}" if label.lower() not in
                                   {"benefit", "what it buys", ""} else val)
                out.append("")
            continue
        out.append(lines[i])
        i += 1
    # collapse the runs of blank lines the substitution can leave behind
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="*", help="tags to fix (default: all releases)")
    ap.add_argument("--check", action="store_true", help="report without writing")
    args = ap.parse_args()

    if args.tags:
        tags = args.tags
    else:
        r = gh("release", "list", "--limit", "100", "--json", "tagName")
        if r.returncode:
            sys.exit(f"gh release list failed: {r.stderr.strip()}")
        tags = [x["tagName"] for x in json.loads(r.stdout)]

    changed = 0
    for tag in tags:
        r = gh("release", "view", tag, "--json", "body")
        if r.returncode:
            print(f"  {tag}: cannot read ({r.stderr.strip().splitlines()[:1]})")
            continue
        body = json.loads(r.stdout).get("body", "") or ""
        new = flatten_tables(body)
        if new.strip() == body.replace("\r\n", "\n").strip():
            print(f"  {tag}: no tables")
            continue
        changed += 1
        if args.check:
            print(f"  {tag}: WOULD flatten {body.count(chr(10) + '|')} table rows")
            continue
        w = gh("release", "edit", tag, "--notes", new)
        print(f"  {tag}: {'rewritten' if not w.returncode else 'FAILED ' + w.stderr.strip()}")

    print(f"\n{changed} release(s) {'would be' if args.check else ''} updated.")


if __name__ == "__main__":
    main()
