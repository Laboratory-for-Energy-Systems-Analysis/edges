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

## Comparison with supplied SimaPro results

The three local `results_*.XLSX` workbooks provide category scores for plastic-film
extrusion, Swiss concrete and Swiss light-commercial-vehicle freight. The reported
method is Ecological Scarcity 2021 V1.06. Their database version is not stated;
18 independent inventory probes support ecoinvent 3.12 cutoff. Long-term emissions
are excluded and Noise is zero in the exports. Concrete's summary uses CH CEM II/A,
but its detailed contribution sheets use ZA CEM II/B-V, so those detail sheets
cannot validate the Swiss concrete flow contributions.

SimaPro is a comparison reference, not an assumed source of truth. The evidence
supports these separate conclusions:

- Three missing carbon-origin aliases are an Edges builder error. Their factors
  (1,000, 28,000 and 30,000 UBP/kg) agree in both usable SimaPro contribution exports
  and the independent ecoinvent 3.12 method. The restored method CSV confirms all
  three factors. The builder aliases and bundled Global warming/total JSON files
  are corrected, rebuilt from the verified source and covered by regression tests.
- Chromium(III) into air receives 230,000 UBP/kg in both Edges and the independent
  ecoinvent method, while the supplied SimaPro exports omit its contribution.
  The restored CSV gives this factor to `Chromium, ion`; FOEN's original Table 40
  explicitly confirms it for chromium(III), supporting retention.
- Nitrogen into surface water receives 36,000 UBP/kg in both Edges and the independent
  ecoinvent method. SimaPro calls the inventory flow `Nitrogen, atmospheric` and
  omits its water-pollutant contribution. The agreement between Edges and ecoinvent
  does not resolve whether this characterization is appropriate: the precise
  substance identity and original method definition need review.
  FOEN Table 45 concerns total nitrogen; this does not establish the inventory
  flow's identity. The restored CSV contains 36,000 UBP/kg for `Nitrogen` and
  `Nitrogen, organic bound` into unspecified water, and zero for groundwater.
  It contains no `Nitrogen, atmospheric` factor. Both nitrogen entries share the
  same CAS number, so that identifier alone does not resolve the substance mapping.
- Edges regionalizes land use by consumer location and derives Noise from transport
  services. The regional land factors come from country-suffixed flow names in the
  source CSV; Edges applies them using the activity location as a proxy for where
  land occupation occurs. These exports are consistent with unregionalized land-flow
  matching and contain no Noise impact. This does not mean that the SimaPro method
  lacks regional factors or cannot apply them.
  These are modelling differences, not sufficient evidence of implementation errors.

Primary reference: [FOEN, Swiss Eco-Factors 2021, Tables 40 and 45 (English PDF)](https://backend.bafu.admin.ch/fileservice/sdweb-docs-prod-bafuadminch-files/files/2025/09/30/3abdc04d-57aa-4328-9264-4170a1d839fc.pdf).
Alternative access: [official publication page](https://www.bafu.admin.ch/en/publication?id=2lirKVJi6lvC).

Tracing the disputed Nitrogen contributions gives 0.227834 UBP for extrusion,
19.794382 UBP for concrete and 0.644466 UBP for transport. The 15 largest emitters
explain 84.64%, 95.06% and 96.43%, respectively. Wastewater markets and onshore/offshore
produced-water discharge are prominent contributors. The raw inventory calculation
reconciles with the existing Edges contribution totals within relative tolerance
`1e-7` for all three cases.

The [ESU petroleum inventory report, Tables 10.4 and 10.5](https://esu-services.ch/fileadmin/download/meili-2023-LCI%20for%20the%20oil%20and%20gas%20extraction.pdf)
includes Nitrogen among produced-water pollutants; it does not establish that this
flow is inert N₂. The report also notes that ecoinvent's implementation can differ
from the submitted data. Accordingly, neither the SimaPro label nor these process
names warrant blanket exclusion. The appropriate substance mapping remains unresolved.

Excluding Noise from comparable totals, 48 of 54 non-negligible category comparisons
are within 1%. An experimental profile can reduce every gap below 0.011% by restoring
the carbon aliases, excluding chromium(III)/nitrogen contributions and applying
unregionalized land factors. This demonstrates which terms explain the differences;
it does **not** justify making all those changes to the bundled method. In particular,
the independently supported carbon fix increases the total-score gap because other
differences previously offset it.

The input workbooks, exploratory comparison scripts, calculation outputs and PDF
reports are local development artifacts. They are not distributed with the package.
The comparison recorded hashes of all 21 method files and checked CF-table and
category/total reconciliation for all three activities. Its wider adjustments
remain diagnostic calculations, not changes to the bundled method.

The restored CSV contained doubled carriage returns. Normalizing only these line
endings recovered the exact reviewed SHA-256 above; no method content was changed.
`review_ubp_carbon_rebuild.py` compares builds with and without the three aliases.
It reproduces every pre-fix method and proves that only 12 carbon rules are added
to Global warming and total. The other 19 category files are byte-identical.
Its compact review is `results/UBP/simapro-method-audit/carbon-rebuild-review.json`.
Run it locally with `python dev/UBP/review_ubp_carbon_rebuild.py`; the generated
review remains outside version control.

The reviewed source-only disposition now contains 1,933 rows (936 non-zero), down
by exactly the three corrected carbon rows. The independent reference gaps decrease
from 96 to 84 signatures. These reviewed sets are hash-pinned, and the regenerated
`ubp_validation_summary.json` passes the full source `build` and `check` commands.

Validation on 11 September 2026 passed 105 fast UBP/core tests and the ecoinvent
3.11 integration test. Fresh runtime audits for 3.10, 3.11 and 3.12 all record the
rebuilt total-method hash and pass matching, reporting and Noise checks. The local
projects currently all use Brightway 2.5 storage. All three SimaPro cases were
recalculated for the 20 categories and total, and the example notebook was rerun
with three embedded figures. These runtime outputs and reports remain local.
