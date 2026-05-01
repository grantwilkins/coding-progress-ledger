# Ledger Observations v0 Summary

Event rows preserve replay fidelity with one row per LedgerEvent prefix. Step rows keep the final state for each (run_id, step) and are intended for plotting and later modeling-oriented analysis.

## Totals

- Total runs: 20
- Event rows: 1388
- Step rows: 704
- Successful runs: 10
- Failed runs: 10
- Unknown success runs: 0

## Category Resolution

Event rows by category resolution mode:

- `native`: 1388

Step rows by category resolution mode:

- `native`: 704

Runs with native/resolved metric mismatch: none

## Non-monotonic Coding Progress

Event-level: `Melevir__cognitive_complexity-15`, `WIPACrepo__iceprod-339`, `asottile__pyupgrade-933`, `asottile__setup-cfg-fmt-132`, `dfm__emcee-510`, `fairlearn__fairlearn-967`, `geomet__geomet-101`, `googleapis__python-spanner-317`, `hsahovic__poke-env-68`, `joke2k__django-environ-174`, `lidatong__dataclasses-json-394`, `mahmoud__boltons-298`, `mc706__changelog-cli-34`, `oasis-open__cti-taxii-client-11`, `omni-us__jsonargparse-370`, `openstack-charmers__zaza-36`, `planetlabs__planet-client-python-389`, `pydantic__pydantic-740`, `python-cmd2__cmd2-681`, `walles__px-50`.

Step-level: `asottile__pyupgrade-933`, `asottile__setup-cfg-fmt-132`, `fairlearn__fairlearn-967`, `openstack-charmers__zaza-36`, `pydantic__pydantic-740`, `walles__px-50`.

## Largest Event-Level Coding Drops

- `Melevir__cognitive_complexity-15`: 0.500000 (investigation)
- `WIPACrepo__iceprod-339`: 0.500000 (investigation)
- `asottile__pyupgrade-933`: 0.500000 (product)
- `asottile__setup-cfg-fmt-132`: 0.500000 (product)
- `dfm__emcee-510`: 0.500000 (investigation)
- `fairlearn__fairlearn-967`: 0.500000 (investigation)
- `geomet__geomet-101`: 0.500000 (product)
- `googleapis__python-spanner-317`: 0.500000 (product)
- `hsahovic__poke-env-68`: 0.500000 (product)
- `joke2k__django-environ-174`: 0.500000 (investigation)

## Largest Step-Level Coding Drops

- `fairlearn__fairlearn-967`: 0.058824 (product)
- `pydantic__pydantic-740`: 0.026316 (investigation)
- `walles__px-50`: 0.025641 (product)
- `asottile__setup-cfg-fmt-132`: 0.017857 (investigation)
- `openstack-charmers__zaza-36`: 0.010989 (product)
- `asottile__pyupgrade-933`: 0.003953 (investigation)

## Largest Event-Level Overall Drops

- `Melevir__cognitive_complexity-15`: 0.500000 (investigation)
- `WIPACrepo__iceprod-339`: 0.500000 (investigation)
- `asottile__pyupgrade-933`: 0.500000 (product)
- `asottile__setup-cfg-fmt-132`: 0.500000 (product)
- `dfm__emcee-510`: 0.500000 (investigation)
- `fairlearn__fairlearn-967`: 0.500000 (investigation)
- `geomet__geomet-101`: 0.500000 (product)
- `googleapis__python-spanner-317`: 0.500000 (product)
- `hsahovic__poke-env-68`: 0.500000 (product)
- `joke2k__django-environ-174`: 0.500000 (investigation)

## Largest Step-Level Overall Drops

- `fairlearn__fairlearn-967`: 0.058824 (product)
- `pydantic__pydantic-740`: 0.026316 (investigation)
- `walles__px-50`: 0.025000 (product)
- `asottile__setup-cfg-fmt-132`: 0.017857 (investigation)
- `openstack-charmers__zaza-36`: 0.010989 (product)
- `asottile__pyupgrade-933`: 0.003937 (investigation)

## Event vs Step

Runs where event-level and step-level largest coding drops differ: `Melevir__cognitive_complexity-15`, `WIPACrepo__iceprod-339`, `asottile__pyupgrade-933`, `asottile__setup-cfg-fmt-132`, `dfm__emcee-510`, `fairlearn__fairlearn-967`, `geomet__geomet-101`, `googleapis__python-spanner-317`, `hsahovic__poke-env-68`, `joke2k__django-environ-174`, `lidatong__dataclasses-json-394`, `mahmoud__boltons-298`, `mc706__changelog-cli-34`, `oasis-open__cti-taxii-client-11`, `omni-us__jsonargparse-370`, `openstack-charmers__zaza-36`, `planetlabs__planet-client-python-389`, `pydantic__pydantic-740`, `python-cmd2__cmd2-681`, `walles__px-50`.

Runs with multiple events at the same step: `Melevir__cognitive_complexity-15`, `WIPACrepo__iceprod-339`, `asottile__pyupgrade-933`, `asottile__setup-cfg-fmt-132`, `dfm__emcee-510`, `fairlearn__fairlearn-967`, `geomet__geomet-101`, `googleapis__python-spanner-317`, `hsahovic__poke-env-68`, `joke2k__django-environ-174`, `lidatong__dataclasses-json-394`, `mahmoud__boltons-298`, `mc706__changelog-cli-34`, `oasis-open__cti-taxii-client-11`, `omni-us__jsonargparse-370`, `openstack-charmers__zaza-36`, `planetlabs__planet-client-python-389`, `pydantic__pydantic-740`, `python-cmd2__cmd2-681`, `walles__px-50`.

## Success / Progress Quadrants

- Success + high progress: `Melevir__cognitive_complexity-15`, `geomet__geomet-101`, `hsahovic__poke-env-68`, `joke2k__django-environ-174`, `lidatong__dataclasses-json-394`, `mahmoud__boltons-298`, `mc706__changelog-cli-34`, `oasis-open__cti-taxii-client-11`, `omni-us__jsonargparse-370`, `planetlabs__planet-client-python-389`.
- Success + low progress: none
- Failure + high progress: `WIPACrepo__iceprod-339`, `asottile__pyupgrade-933`, `asottile__setup-cfg-fmt-132`, `dfm__emcee-510`, `fairlearn__fairlearn-967`, `googleapis__python-spanner-317`, `openstack-charmers__zaza-36`, `pydantic__pydantic-740`, `python-cmd2__cmd2-681`, `walles__px-50`.
- Failure + low progress: none
- Unknown success: none

## Sanity Check Warnings

- none
