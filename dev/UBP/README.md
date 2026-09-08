# Ecological Scarcity 2021 (UBP) build

`build_ubp_methods.py` converts the reviewed SimaPro 10.4 CSV export of
Ecological Scarcity 2021 version 1.06 into the built-in Edges methods.

The source file is licensed and intentionally ignored by Git. Place it at
`dev/UBP/Ecological-Scarcity-21.CSV`; its expected SHA-256 is
`20e46d0acb9e2a676f35636e76ae0ccacacdae122e49459985608a290525147a`.

The audit requires local Brightway projects named `ecoinvent-3.10-cutoff`,
`ecoinvent-3.11-cutoff`, and `ecoinvent-3.12-cutoff`, with the database names
declared in `DEFAULT_TARGETS`. Catalogs are opened read-only.

```console
python dev/UBP/build_ubp_methods.py validate
python dev/UBP/build_ubp_methods.py build
python dev/UBP/build_ubp_methods.py check
```

`build` writes 20 category JSON files, the total JSON file, and
`ubp_validation_summary.json`. `check` rebuilds everything in memory and fails
if any committed output or the audit summary is stale.

Validation covers source identity and structure, complete row disposition,
exact ecoinvent supplier reachability both per version and across the union,
the independently pinned ecoinvent 3.12 reference-method overlap, Edges
signature uniqueness, and all six Noise transport families. Source rows which
have no compatible flow in the union of the three target versions are retained
in the summary as explicit source-only groups. Their canonical hash is pinned,
so any change to this reviewed disposition fails validation.

## Runtime matching audit

`run_ubp_matching_audit.py` performs end-to-end Edges calculations with the
cutoff projects for ecoinvent 3.10, 3.11, and 3.12. It uses one exact functional
unit in every version: 1 ton kilometer of the CH market for transport, freight,
lorry 28 metric ton, fatty acid methyl ester 100%. This functional unit was
chosen because its supply chain contains elementary flows and a guaranteed
direct technosphere Noise exchange. A second exact passenger-rail market probe
ensures that the rail-passenger Noise family is exercised as well.

```console
python dev/UBP/run_ubp_matching_audit.py --versions 3.12
python dev/UBP/run_ubp_matching_audit.py --versions 3.11 3.12
python dev/UBP/run_ubp_matching_audit.py --versions 3.12 --full-cf-table
```

`--versions` is required so that the selected projects are explicit and use a
Brightway-compatible Python environment. The script writes compact JSON and CSV
summaries plus matched-only Noise tables under `results/UBP/`. Pass
`--full-cf-table` to additionally write gzip-compressed total-method tables,
including unmatched inventory exchanges. Detailed CF tables contain licensed,
calculation-specific ecoinvent metadata and are ignored by Git. Runs for a
subset of versions update those versions while retaining existing results for
the others; each result records its own timestamp, Git revision, method hashes,
Python/Brightway/Edges versions, project storage generation, and warnings.

The audit fails if:

- a total-method rule is ignored or has an effective duplicate signature;
- an inventory exchange directly eligible for a UBP elementary-flow or Noise
  rule remains unmatched;
- an exchange position is characterized twice, or matched/unmatched positions
  do not form an exact partition of the assessed inventory;
- the CF-table impact sums do not reconcile with the matrix and total scores;
- the total method's technosphere component differs from a separate Noise run;
- the direct 28-tonne lorry edge is absent or does not receive the reviewed
  `180 / 5.79` UBP/tkm factor;
- the passenger-rail probe is absent or does not receive 8 UBP/person-km;
- per-strategy match counts do not reconcile with the final matched positions.

The three local projects can use different Brightway storage generations. Run
each with a Brightway environment compatible with its project format; the audit
opens projects read-only and never migrates or modifies their databases.
