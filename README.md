# GECS: A Graph-Based Echo Chamber Score

Code and queries to reproduce the results in *"GECS: A Graph-Based Echo
Chamber Score for Structural Detection of Ideological Isolation on
Twitter/X."*

GECS is a content-agnostic metric that scores retweet communities on four
structural dimensions — **Isolation**, **Internal Density**, **Deafness**,
and **Fixation** — computed directly from a weighted, directed retweet
graph, with no reliance on text, metadata, or ground-truth labels.

This pipeline is built on **Neo4j** and **Cypher**: the retweet graph is
constructed and analyzed inside Neo4j using the Graph Data Science (GDS)
library, and the resulting per-community scores are exported to JSON and
turned into the paper's figures with a small Python script.

## Repository contents

| File | Description |
|---|---|
| [`CYPHER_GUIDE.md`](CYPHER_GUIDE.md) | Step-by-step Cypher/GDS queries to build the retweet graph, detect communities, and compute all four GECS dimensions in Neo4j. |
| `generate_figures.py` | Reads the exported JSON results and generates every figure used in the paper. |

## Requirements

- **Neo4j** (Desktop or Server) with the [**APOC**](https://neo4j.com/labs/apoc/) and [**Graph Data Science (GDS)**](https://neo4j.com/docs/graph-data-science/current/installation/) plugins installed
- **Python 3.x** with:
  ```bash
  pip install matplotlib numpy
  ```

## Reproducing the results

1. **Load your retweet data into Neo4j** as `User` and `Message` nodes, connected by `(:User)-[:CREATED]->(:Message)` and `(:Message)-[:DERIVED_BY]->(:Message)` for retweet links (see the prerequisite note in `CYPHER_GUIDE.md`).
2. **Run the Cypher queries in [`CYPHER_GUIDE.md`](CYPHER_GUIDE.md) in order** to:
   - build the weighted retweet graph (Step 1),
   - project it into GDS and detect communities (Steps 2–3),
   - compute the four GECS dimensions plus supporting diagnostics (Steps 4–12).
3. **Export the per-community results to JSON**, matching the three files `generate_figures.py` expects:
   - `final_cecs_scores.json` — per-chamber GECS + dimension scores
   - `stance_each_community.json` — per-user political-stance data
   - `Generall_info.json` — overall corpus statistics
4. **Generate the figures:**
   ```bash
   python generate_figures.py
   ```
   or, to plot a specific set of chambers instead of the default list:
   ```bash
   python generate_figures.py 22493 31969 17287
   ```
   Figures are saved as PNGs in `output_figures/`.

## Dataset
You can find dataset here :
https://www.kaggle.com/datasets/javadhamidzadeh/gecs-data

## Citation

If you use this code or dataset, please cite the paper (full citation to be
added once published).

## Data availability

The retweet dataset used in this study is available from the corresponding
author upon reasonable request for academic research purposes.


