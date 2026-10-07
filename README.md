# Job Ecosystem

A personal job-hunting pipeline in one package: scrape job boards into SQLite,
fetch and embed the full descriptions, triage in a terminal UI, and tailor your
resume to a posting.

    scrape  ->  describe  ->  embed  ->  triage  ->  tailor

## Install

Python 3.11 or newer. Heavy dependencies are optional extras; the scrape and
query paths need only `httpx`.

    python -m venv .venv
    .venv/bin/pip install -e .                    # core: scraping, store, queries
    .venv/bin/pip install -e ".[tui]"             # terminal UI
    .venv/bin/pip install -e ".[embed]"           # Voyage embeddings
    .venv/bin/pip install -e ".[pdf]"             # resume PDF rendering
    .venv/bin/pip install -e ".[tui,embed,pdf]"   # everything

## The commands

| command | what it does |
|---|---|
| `jobecosystem-scrape` | fetch listings from every configured board into the database |
| `jobecosystem-describe` | fetch full descriptions for listings that lack one (paced, resumable) |
| `jobecosystem-embed` | embed descriptions with Voyage for similarity search |
| `jobecosystem-discover` | grow the company lists from Hacker News "Who is hiring" links |
| `jobecosystem-triage` | browse and triage jobs in a terminal UI |
| `jobecosystem-tailor` | validate, propose, diff, and render a tailored resume |

### Scrape

    jobecosystem-scrape                     # all sources
    jobecosystem-scrape --source lever      # one family (repeatable)
    jobecosystem-scrape --dry-run           # list the sources, then exit
    jobecosystem-scrape --quiet             # exit code only

Sources are Ashby, SmartRecruiters, Greenhouse, Lever, Workday, and the Workable
marketplace. Each board is a line in its file (below); Workable needs no file —
it reads Workable's cross-company feed, capped by `--workable-pages`.

Exit code is 0 when every source finished, 1 when some failed (the rest still
ran), 2 when nothing could start.

### Describe

Listings carry metadata only, so the full text is fetched later, one paced
request per job. The queue is persistent: re-running resumes where the last run
stopped.

    jobecosystem-describe --limit 300
    jobecosystem-describe --limit 300 --delay 1.0 --jitter 0.5   # gentler

Ashby, Lever, and Workable listings include the text, so those rows never enter
the queue.

### Embed

Needs the `embed` extra and a `VOYAGE_API_KEY`. Only described rows are eligible.

    VOYAGE_API_KEY=... jobecosystem-embed --limit 500

`--model` defaults to Voyage's; changing it re-embeds rows written by another
model. `--force` re-embeds rows already on the current model.

### Discover

Reads the current "Ask HN: Who is hiring?" thread and appends any ATS company
slugs it finds to the matching `*_companies.txt` / `*_boards.txt` file. Review
the diff before the next scrape.

    jobecosystem-discover --dry-run         # print what would be added
    jobecosystem-discover                   # append it
    jobecosystem-discover --family lever

### Triage

    jobecosystem-triage                     # the default database
    jobecosystem-triage --db /tmp/jobs.db
    jobecosystem-triage --list-sources      # print counts, then exit

Keys: `/` search · `]` `[` next/previous tab · `s` seen · `a` applied · `z` new ·
`x` hide · `ctrl+0`–`ctrl+5` rate · `d` fetch description · `c` copy URL ·
`ctrl+e` more like this · `ctrl+r` match your resume · `r` reload · `q` quit.

### Tailor

**The tailor section is a work in progress, only search is reliable at the moment.**

    jobecosystem-tailor search --limit 25                       # jobs like the resume
    jobecosystem-tailor propose --job-id 123 --out overlay.json # overlay, no PDF
    jobecosystem-tailor diff --overlay overlay.json             # review it
    jobecosystem-tailor render --overlay overlay.json --approve --out resume.pdf
    jobecosystem-tailor validate --overlay overlay.json

`search` needs the `embed` extra and a key; `render` needs the `pdf` extra and
refuses to write a PDF without `--approve` (exit 3). `propose` shells out to the
command in `TAILOR_COMPLETE_COMMAND` (prompt on stdin, JSON overlay on stdout)
and needs no built-in model. An overlay may only select, reorder, and rephrase
the base resume's facts — it can never add new ones.

## Configuration

The database is `--db`, else `$DB_PATH`, else `jobs.db` at the repo root.

One board per line in these files at the repo root; each is overridable by an
environment variable or the matching `--...` flag:

| file | source | environment variable |
|---|---|---|
| `ashby_boards.txt` | Ashby | `ASHBY_BOARDS_FILE` |
| `smartrecruiters_companies.txt` | SmartRecruiters | `SMARTRECRUITERS_COMPANIES_FILE` |
| `greenhouse_companies.txt` | Greenhouse | `GREENHOUSE_BOARDS_FILE` |
| `lever_companies.txt` | Lever | `LEVER_COMPANIES_FILE` |
| `workday_tenants.txt` | Workday | `WORKDAY_TENANTS_FILE` |
| — | Workable marketplace | — |

Other environment variables:

| variable | used by | meaning |
|---|---|---|
| `VOYAGE_API_KEY` | embed, triage, tailor search | Voyage API key |
| `TAILOR_RESUME_FILE` | tailor | base resume (default `resume.base.json`) |
| `TAILOR_LAYOUT_FILE` | tailor render | JSON overrides for the PDF layout |
| `TAILOR_COMPLETE_COMMAND` | tailor propose | LLM command: prompt on stdin, JSON on stdout |

## Scheduling

`how_to_cron.txt` covers crontab entries, exit codes, and how to verify a run
from the database. `scripts/run-pipeline.sh` runs scrape → describe → embed in
order, with a timestamped line before and after each step, and a dated log file.

## Tests

    .venv/bin/python -m pytest

## Layout

    src/jobecosystem/core      database, migrations, models, embeddings, similarity
    src/jobecosystem/ingest    scrapers, description fetch, embedding, discovery, CLI
    src/jobecosystem/triage    query layer and the Textual TUI
    src/jobecosystem/tailor    structured resume, overlay, validation, PDF rendering
