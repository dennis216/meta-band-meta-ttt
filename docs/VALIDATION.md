# Publication validation

This record covers code packaging, not repeated training or verification of paper performance.

Release checks used Python 3.11 on Linux (WSL2). Installed versions are recorded in `environment/tested-versions.txt`.

| Check | Result |
|---|---|
| Existing TUSZ tests, data, evaluation, preprocessing, and contracts | 134 passed, 1 warning |
| Publication and relocation regression tests | 4 passed |
| Python AST syntax checks | 352 files passed |
| Tests in a clean export of the Git index | 138 passed, 1 warning |
| Complete runtime-copy relocation | Paths replaced in 62 files; data-processing packages retained |
| Relocated entry-point smoke checks | `--help` passed for TUSZ data preparation, v4 training, and CHB Meta training |
| Raw-data, weight, and common credential-pattern scan | No prohibited artifacts found |
| Router / idea3 experiment entry points | Excluded |

Verification command:

```bash
python tools/verify.py
```

Checks do not download EEG or weights, run full training, repeat GPU parallel-equivalence tests, or cover every historical model test. Some `tests/models` cases require unpublished checkpoints or other model dependencies. The 138 passing tests therefore do not represent the entire test directory.

Correction on 2026-10-02: the initial `.gitignore` rule `data/` excluded `src/bfa/data` and `tests/data` unintentionally. The relocation tool used the same overly broad directory filter. Both now exclude only root-level data directories, and the eight missing source/test files were added to Git. Verification used a clean directory exported with `git checkout-index`, without relying on uncommitted working-tree files. Release checks also verify that files listed in the source manifest exist. The initial 135-test run used the complete local copy and did not establish completeness of the initial remote repository.

The remaining warning comes from an existing truncated-gradient test converting a requires-grad tensor to a scalar for an assertion. The test passes. Research algorithms were not changed to suppress the warning.

Source files retain the research implementation wherever possible. `docs/source-manifest.json` records SHA-256 hashes before copying; packaging documentation, tools, dependencies, and tests are maintained separately. Historical absolute paths remain in source and are relocated by the runtime-copy tool. Shared historical model adapters were not rewritten as new methods.

The CHB external implementation comes from the frozen 2026-09-05 bundle; TUSZ comes from the research checkout. Archived and current entry points may use different metric semantics, so their results cannot simply be combined. Exact reproduction still requires the original patient splits, cache versions, and checkpoints, which are not publicly distributed.

English-only update on 2026-10-03: documentation, comments, docstrings, and human-readable report/error messages were translated. Original source hashes remain provenance records, not hashes of the translated files. Numerical algorithms, identifiers, and experiment settings are unchanged.

After translation, all 395 tracked files were scanned for Chinese text with no matches. AST comparisons of the 13 edited Python files confirmed that only translated text constants changed beyond comments. Release checks passed, and the verification suite again reported 138 passed with the same existing warning.
