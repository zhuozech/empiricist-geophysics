# Knowledge map: inverse problems and Bayesian inversion

A structured map of the ideas in Topic 2 of *Empiricist Geophysics*. Each idea is a small topic with a plain
description, what you should be able to do once you have it, a self-check question, the notebook section that
introduces it, and the topics it builds on. The format follows the
[Marble skill taxonomy](https://github.com/withmarbleapp/os-taxonomy): topics, prerequisite edges, clusters and a
manifest, as plain JSON with schemas and a validator.

- **Explore it**: `html/knowledge-map.html`, a rotating 3-D map after Marble's curriculum view. Every dot is a topic,
  colour is its section, height is reading order (notebook 2.01 at the bottom, 2.33 at the top) and every thread is a
  prerequisite. Click a dot to trace everything it builds on, all the way back; the card links to the notebook section.
  Search finds a topic by name, and `knowledge-map.html#eg_cholesky` opens one topic directly.
- **See it at once**: `figures/knowledge-map.png`, a 300 dpi still of the same view.

<!-- counts -->
**223 topics** from 33 notebooks, **381 prerequisite links** (334 hard, 47 soft), 7 sections.

| section | topics |
|---|---|
| A · Probability for inverse problems | 54 |
| B · Linear inversion, ill-posedness, priors | 48 |
| C · Errors and data weighting | 12 |
| D · Computing a posterior: sampling | 47 |
| E · Time-dependent and ensemble inversion | 22 |
| F · Hyperparameters and weights | 30 |
| G · Checking and synthesis | 10 |

By type: conceptual 110, procedural 64, representational 10, language 3, meta 36.
<!-- /counts -->

## Files

```
data/topics.json         the topics
data/dependencies.json   prerequisite edges: topicId depends on prerequisiteId
data/clusters.json       one summary per section (A-G) of Topic 2
data/manifest.json       counts and SHA-256 checksums
schema/*.schema.json     JSON Schema (draft 2020-12) for the three data files
source/                  the hand-written source: one T(...) entry per topic, prerequisites listed in the topic
scripts/build.py         source -> data (adds the computed fields)
scripts/validate.py      integrity checks (exit code 1 on any error)
scripts/build_viewer.py  data -> 3-D layout -> html/knowledge-map.html and figures/knowledge-map.png
```

## Topic fields

| field | meaning |
|---|---|
| `id` | stable identifier, `eg_` plus a readable name |
| `type` | `CONCEPTUAL` (an idea), `PROCEDURAL` (a method you carry out), `REPRESENTATIONAL` (a picture you learn to read), `LANGUAGE` (a vocabulary distinction), `META` (a judgement: checking, choosing, reporting) |
| `subject`, `domain` | Topic 2 of the course, and its section A-G |
| `name`, `description` | the topic in plain words |
| `introducedIn`, `section`, `url` | the notebook and section that introduce it, and a link to that section |
| `evidence` | what you should be able to do once you have it |
| `assessmentPrompt` | a self-check question |
| `figure` | a key figure in `figures/` |
| `usedIn` | computed: later notebooks whose text mentions the topic |
| `mentionedBefore` | computed: earlier notebooks that mention it before it is introduced (previews) |
| `centrality` | computed: the share of all topics that build on it, directly or through others, scaled so the largest is 1 |

Each edge has a `strength`: `hard` (needed first) or `soft` (helpful), and a one-line `reason`.

## The 3-D layout

`build_viewer.py` computes every position once, so the page only rotates and draws. Height follows the reading order
(by notebook, then section). The radius grows with height at the rate of Marble's view (107 + 0.25 × height in layout
units, a straight-line fit to their positions), times a random factor drawn from a normal distribution with mean 1 and
standard deviation 0.17, clipped to 0.68–1.32, so the map widens into a funnel with some depth. The angle around the axis comes from a
force relaxation in which all dots repel and linked topics attract, more strongly when they sit at about the same
height, so that links stay short. The random numbers use a fixed seed, so a rebuild gives the same picture.

`usedIn` and `mentionedBefore` come from matching the topic's specific words in the notebook text, so they are a
guide, not a guarantee. `validate.py` additionally checks that the prerequisite graph has no cycles and that every hard
prerequisite comes earlier in the reading order than the topic that needs it.

## Rebuilding

```bash
python knowledge-map/scripts/build.py        # source -> data; warns if a topic's words are missing from its section
python knowledge-map/scripts/validate.py     # must print "OK: all checks passed"
python knowledge-map/scripts/build_viewer.py # interactive page and static figure
```

Run them after editing the notebooks or the source, so section links and computed fields stay current.

## License

The data and text are released under CC BY 4.0, like the rest of the course text (`../LICENSE-CONTENT.md`); the
scripts under MIT (`../LICENSE`).
