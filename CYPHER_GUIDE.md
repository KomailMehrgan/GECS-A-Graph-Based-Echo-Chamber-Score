# Cypher Query Guide — Echo Chamber Detection (GECS)

This document contains the Neo4j / Cypher queries used to build the retweet
graph, detect communities, and compute the structural metrics behind the
**GECS** (Graph-based Echo Chamber Score) paper.

Run the queries in order in the Neo4j Browser, `cypher-shell`, or the Neo4j
Python driver. All community-detection and scoring steps require the
[Neo4j Graph Data Science (GDS) library](https://neo4j.com/docs/graph-data-science/current/installation/)
and the [APOC library](https://neo4j.com/labs/apoc/) to be installed on your
Neo4j instance.

**Prerequisite data model:** the queries assume `User` and `Message` nodes
already loaded, where `Message` nodes carry a `msg_type` property
(`1` = original tweet, `2` = retweet), connected by `(:User)-[:CREATED]->(:Message)`
and `(:Message)-[:DERIVED_BY]->(:Message)` (a retweet message pointing to the
message it retweeted).

---

## Step 1: Network Construction (the retweet graph)

**Objective:** condense the retweet chain into direct, weighted user-to-user
edges.

**Method:** `apoc.periodic.iterate` batches the operation so it doesn't
exhaust memory on large corpora.

```cypher
CALL apoc.periodic.iterate(
  // Phase 1: drive the batches using the retweet messages
  "MATCH (m1:Message {msg_type: 2}) RETURN m1",

  // Phase 2: trace the path to the original author
  "MATCH (u1:User)-[:CREATED]->(m1)-[:DERIVED_BY]->(m2:Message {msg_type: 1})<-[:CREATED]-(u2:User)
   MERGE (u1)-[rel:RETWEETED]->(u2)
   ON CREATE SET rel.weight = 1
   ON MATCH SET rel.weight = rel.weight + 1",

  // Configuration: process 10,000 retweets per batch
  {batchSize: 10000, parallel: false}
)
```

---

## Step 2: Algorithmic Preparation (in-memory GDS projection)

**Objective:** load the graph into an in-memory GDS workspace.

**Method:** project as `UNDIRECTED` (required for community detection) and
carry the `weight` property along.

```cypher
CALL gds.graph.project(
  'retweetGraph',
  'User',
  {
    RETWEETED: {
      orientation: 'UNDIRECTED',
      properties: 'weight'
    }
  }
)
YIELD graphName, nodeCount, relationshipCount;
```

---

## Step 3: Community Detection

**Objective:** group users into structural factions based on mutual
amplification.

> **Note:** the query below runs Louvain. If you are reproducing the paper's
> reported communities, replace this with `gds.leiden.write` (the paper's
> algorithm of record — see the note at the end of this guide).

```cypher
CALL gds.louvain.write('retweetGraph', {
  relationshipWeightProperty: 'weight',
  writeProperty: 'COM_ID'
})
YIELD communityCount, modularity, ranLevels;
```

---

## Step 4: Macro-Community Identification

**Objective:** isolate the largest communities driving the echo-chamber
effect.

```cypher
MATCH (u:User)
WHERE u.COM_ID IS NOT NULL
RETURN u.COM_ID AS Echo_Chamber_ID,
       count(u) AS Total_Users_In_Chamber
ORDER BY Total_Users_In_Chamber DESC
LIMIT 10;
```

---

## Step 5: Visual Topology Sampling

**Objective:** extract a manageable, representative sub-graph of the largest
factions for visualization.

```cypher
// 1. Find the top 5 biggest echo chambers
MATCH (u:User)
WHERE u.COM_ID IS NOT NULL
WITH u.COM_ID AS chamber_id, count(u) AS chamber_size
ORDER BY chamber_size DESC
LIMIT 5

// 2. Grab a representative sample of connections from inside each chamber
CALL {
  WITH chamber_id
  MATCH (u1:User {COM_ID: chamber_id})-[r:RETWEETED]->(u2:User {COM_ID: chamber_id})
  RETURN u1, r, u2
  LIMIT 10000
}
RETURN u1, r, u2;
```

---

## Step 6: Isolation Metric (internal vs. external retweets)

**Objective:** calculate the ratio of internal retweets to cross-community
retweets — the raw ingredient for the **Isolation** dimension.

```cypher
MATCH (u1:User)-[r:RETWEETED]->(u2:User)
WHERE u1.COM_ID IN [/* your chamber IDs here */]
WITH u1.COM_ID AS Chamber_ID,
     sum(CASE WHEN u1.COM_ID = u2.COM_ID THEN r.weight ELSE 0 END) AS Internal_Retweets,
     sum(CASE WHEN u1.COM_ID <> u2.COM_ID THEN r.weight ELSE 0 END) AS External_Retweets
RETURN Chamber_ID,
       Internal_Retweets,
       External_Retweets,
       round((toFloat(Internal_Retweets) / (Internal_Retweets + External_Retweets)) * 100, 2) AS Internal_Percentage
ORDER BY Internal_Percentage DESC;
```

---

## Step 7: E/I Index (insularity score)

**Objective:** quantify external vs. internal interaction using the classic
External–Internal Index.

```cypher
MATCH (u1:User)-[r:RETWEETED]->(u2:User)
WHERE u1.COM_ID IN [/* your chamber IDs here */]
WITH u1.COM_ID AS Chamber_ID,
     sum(CASE WHEN u1.COM_ID = u2.COM_ID THEN r.weight ELSE 0 END) AS I,
     sum(CASE WHEN u1.COM_ID <> u2.COM_ID THEN r.weight ELSE 0 END) AS E
RETURN Chamber_ID,
       round(toFloat(E - I) / (E + I), 3) AS Insularity_Score
ORDER BY Insularity_Score ASC;
```

---

## Step 8: Internal Density (local clustering coefficient)

**Objective:** measure how tightly interconnected users are within each
community — the **Internal Density** dimension.

```cypher
CALL gds.localClusteringCoefficient.stream('retweetGraph')
YIELD nodeId, localClusteringCoefficient
WITH gds.util.asNode(nodeId) AS u, localClusteringCoefficient
WHERE u.COM_ID IN [/* your chamber IDs here */]
RETURN u.COM_ID AS Chamber_ID,
       avg(localClusteringCoefficient) AS Internal_Density
ORDER BY Internal_Density DESC;
```

---

## Step 9: Conductance ("leakage" test)

**Objective:** measure the probability of information escaping the
community — ratio of external edges (cut) to total activity (volume).

```cypher
MATCH (u1:User)-[r:RETWEETED]->(u2:User)
WHERE u1.COM_ID IN [/* your chamber IDs here */]
WITH u1.COM_ID AS Chamber_ID,
     sum(CASE WHEN u1.COM_ID <> u2.COM_ID THEN r.weight ELSE 0 END) AS External_Edges,
     sum(r.weight) AS Total_Volume
RETURN Chamber_ID,
       External_Edges AS Cut,
       Total_Volume AS Volume,
       round(toFloat(External_Edges) / Total_Volume, 4) AS Conductance
ORDER BY Conductance ASC;
```

---

## Step 10: Bridge Users ("Diplomats")

**Objective:** count the share of each community's population that interacts
externally (i.e., acts as a bridge to other communities).

```cypher
// 1. First, count the total population of each chamber
MATCH (u1:User)
WHERE u1.COM_ID IN [/* your chamber IDs here */]
WITH u1.COM_ID AS Chamber_ID, count(u1) AS Total_Population

// 2. Then, find the specific users ("diplomats") interacting externally
MATCH (u2:User)-[:RETWEETED]->(u3:User)
WHERE u2.COM_ID = Chamber_ID AND u3.COM_ID <> Chamber_ID
WITH Chamber_ID, Total_Population, count(DISTINCT u2) AS Diplomats

// 3. Return the percentage of the population acting as bridges
RETURN Chamber_ID,
       Total_Population,
       Diplomats,
       round((toFloat(Diplomats) / Total_Population) * 100, 2) AS Diplomat_Percentage
ORDER BY Diplomat_Percentage ASC;
```

---

## Step 11: Deafness (Bridge Concentration Score)

**Objective:** measure how much internal influence is concentrated in bridge
("diplomat") users versus the community as a whole — the **Deafness**
dimension.

```cypher
MATCH (u:User)
WHERE u.COM_ID IN [/* your chamber IDs here */]

// 1. Identify diplomats (users with external interactions)
OPTIONAL MATCH (u)-[r_ext:RETWEETED]->(ext:User)
WHERE u.COM_ID <> ext.COM_ID
WITH u, count(r_ext) > 0 AS is_diplomat

// 2. Calculate internal influence (audience size)
OPTIONAL MATCH (peer:User)-[r_in:RETWEETED]->(u)
WHERE peer.COM_ID = u.COM_ID
WITH u, u.COM_ID AS Chamber_ID, is_diplomat, count(r_in) AS internal_influence

// 3. Calculate the bounded Bridge Concentration Score
RETURN Chamber_ID,
       sum(CASE WHEN is_diplomat THEN 1 ELSE 0 END) AS Total_Diplomats,
       count(u) AS Total_Population,
       sum(CASE WHEN is_diplomat THEN internal_influence ELSE 0 END) AS Sum_Bridge_Influence,
       sum(internal_influence) AS Total_Internal_Influence,
       round(
         toFloat(sum(CASE WHEN is_diplomat THEN internal_influence ELSE 0 END))
         / (sum(internal_influence) + 0.0001), 4
       ) AS Bridge_Concentration_Score
ORDER BY Bridge_Concentration_Score DESC;
```

---

## Step 12: Fixation (neighborhood entropy)

**Objective:** measure whether a community's external attention is
concentrated on one neighboring community or spread across several — the
**Fixation** dimension, via Shannon entropy.

```cypher
// 1. Group retweets by target community
MATCH (u1:User)-[r:RETWEETED]->(u2:User)
WHERE u1.COM_ID_Leiden IN [/* your chamber IDs here */]
WITH u1, u1.COM_ID_Leiden AS Chamber_ID, u2.COM_ID_Leiden AS target_com, count(r) AS weight_to_target

// 2. Aggregate per user, storing per-target weights in a temporary list
WITH u1, Chamber_ID, sum(weight_to_target) AS total_user_weight,
     collect({target: target_com, weight: weight_to_target}) AS targets

// 3. Unwind the list back into rows
UNWIND targets AS t

// 4. Calculate probability (p_ic)
WITH u1, Chamber_ID, toFloat(t.weight) / total_user_weight AS p_ic

// 5. Calculate individual user entropy (H_i), log base 2
WITH u1, Chamber_ID, sum(-p_ic * (log(p_ic) / log(2.0))) AS H_i

// 6. Average the entropy per community (raw, unnormalized H_c)
RETURN Chamber_ID,
       avg(H_i) AS H_c
ORDER BY H_c ASC;
```

---

## Important note before you run these end-to-end

Steps 6–11 use `u.COM_ID` (written by the **Louvain** call in Step 3) and
placeholder chamber-ID lists left over from an earlier exploratory run.
Step 12 uses `u.COM_ID_Leiden` and the ten chamber IDs actually reported in
the paper (`17381, 22493, 31969, 17287, 21579, 9053, 17192, 1000, 27903,
17526`). **These are inconsistent as written** — the paper's methodology
section specifies the **Leiden** algorithm throughout, not Louvain.




Once each step above has been run for your chosen community IDs, export the
per-chamber results into the three JSON files consumed by
`generate_figures.py` (see the main [README](README.md)):
`final_cecs_scores.json`, `stance_each_community.json`, and
`Generall_info.json`.
