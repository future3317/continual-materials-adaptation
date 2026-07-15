# Repository & Dependency Audit Report

Project root: `E:/CODE/Continual Learning`  
Date: 2026-07-15  
Auditor: Kimi Code CLI

---

## Part A — GitHub Repository Audit

### 1. Remote, branches, and reachability

| Item | Value |
|------|-------|
| Remote URL | `https://github.com/future3317/continual-materials-adaptation.git` |
| Branches (local) | `main` |
| Branches (remote) | `remotes/origin/main`, `remotes/origin/HEAD` |
| `git fetch origin --dry-run` | Exit 0, reachable. No new objects to fetch. |
| Latest commit on **origin/main** | `209ebcb25cb560b34ab831aa5429c598b2b0ef5f` — 2026-07-15 13:53:41 +0800, "Add ALIGNN backbone (pure-PyTorch, no DGL)" |
| Latest commit on **local main** | `ea6a71ee6e9cc3b46ca9861336b16c94c01ee415` — 2026-07-15 15:01:02 +0800, "refactor(adapters): make SingleChildTucker a semantic alias of LoRA-ABA" |
| Relative status | **Local `main` is 1 commit ahead of `origin/main`**. No uncommitted changes. No stashes. |

### 2. Commit history cleanliness (ICLR readiness)

```text
* ea6a71e (HEAD -> main) refactor(adapters): make SingleChildTucker a semantic alias of LoRA-ABA
* 209ebcb (origin/main) Add ALIGNN backbone (pure-PyTorch, no DGL)
* 1634319 Implement 反馈_2.md P0/P1/P2 upgrades
* 3334349 Initial commit: FR-PhyTCA continual materials adaptation codebase
```

**Assessment:** The history is short but clean enough for an ICLR code release:

- Each commit is a single logical milestone.
- Messages are concise and describe *what* and *why*.
- No merge-conflict artifacts, no WIP/temp commits, no binaries in Git history.
- Author metadata is consistent (`Researcher <researcher@example.com>`).

**Before submission, consider:**

1. Squashing/rewording the Chinese-character commit (`反馈_2.md`) to English for reviewer accessibility, or keeping it but adding an annotated tag/release note explaining it.
2. Adding a `LICENSE` file.
3. Replacing the generic author email with a real contact address.
4. Making sure `README.md` is accurate (it already is; see Part B note about dependencies).

### 3. Sensitive files and large-file scan

- **Sensitive files searched:** `.env*`, `*.pem`, `*.key`, `*.crt`, `credentials*`, `*.secret`, `id_rsa*`
- **Result:** None found in the working tree.
- **Tracked files:** 33 files total, all source/config/test files.
- **Large files present locally (but excluded by `.gitignore`):**
  - `artifacts/init/seed_42_base.pt`
  - `artifacts/init/seed_43_base.pt`
  - `artifacts/init/seed_44_base.pt`
  - `reports/manifest_protocol_a.json`
  - `reports/manifest_protocol_b.json`
  - `reports/phase0_b_screening/opt_parent_bundle.pt`
  - `reports/phase2_b_scaling/opt_parent_bundle.pt`
  - `reports/phase2_smoke/opt_parent_bundle.pt`

These are correctly ignored by `.gitignore` (`artifacts/`, `reports/`, `*.pt`, `*.pth`, `*.ckpt`, `data_cache/`, `logs/`, etc.). They are **not tracked** and therefore will not be pushed. Good.

### 4. Is it safe to delete/replace the old GitHub repo?

**Current state:**

- `origin` points to `future3317/continual-materials-adaptation`.
- The remote is reachable and has three commits on `main`.
- Local `main` has one additional commit not yet pushed (`ea6a71e`).
- No local branches other than `main`; no stashes; no uncommitted work.

**Conclusion:** The old repo **can be safely deleted/replaced** *after* the local-only commit is preserved. The remote currently lacks `ea6a71e`, so if you delete the GitHub repo before pushing, you would lose ~1 hour of work.

**Recommended safe sequence:**

1. Do **not** delete the old repo yet.
2. Create the new GitHub repo with one of the names below.
3. Add the new repo as a second remote, e.g.:
   ```bash
   git remote add new-origin https://github.com/<user>/<new-name>.git
   git push new-origin main
   ```
4. Only after verifying the new remote contains `ea6a71e` and all history, delete the old repo.

Alternatively, force-push `main` to the *existing* remote if you want to keep the URL. I did **not** push.

### 5. Repository name recommendations

Rationale: The paper/method is **FR-PhyTCA** — Fidelity-Residual Physics-Structured Tensor Component Adaptation for continual learning on materials databases. The repository name should be discoverable, concise, and not conflict with common library names.

| # | Proposed name | Rationale |
|---|---------------|-----------|
| 1 | `fr-phytca` | Clean, paper-method-first, easy to cite. Best default. |
| 2 | `continual-materials-adaptation` | Descriptive; matches current repo concept but shorter and without the typo-prone "continual-materials-adaptation" exact wording. |
| 3 | `phytca` | Short and brand-like. Risk: ambiguous without context. |
| 4 | `fr-phytca-continual` | Adds "continual" keyword for searchability; slightly longer but very clear. |
| 5 | `continual-learning-materials` | Domain-first; good if you plan to broaden beyond FR-PhyTCA later. |

**Top recommendation:** `fr-phytca` (or `fr-phytca-continual` if you want extra discoverability).

---

## Part B — Dependency Status

### 1. Import checks

| Package | Import result | Version / error |
|---------|---------------|-----------------|
| `alignn` | ✅ OK | `2026.5.20` |
| `dgl` | ❌ FAIL | `FileNotFoundError: Cannot find DGL C++ graphbolt library at ...\graphbolt_pytorch_2.12.0.dll` |
| `torch` | ✅ OK | `2.12.0+cu126` |

### 2. DGL failure explanation

The DGL wheel installed in the `EGNN` environment was built for a PyTorch ABI that does not match PyTorch 2.12.0. DGL attempts to lazy-load `graphbolt_pytorch_2.12.0.dll` and the file is missing, so `import dgl` raises `FileNotFoundError`.

This is a known DGL/PyTorch-version compatibility issue. The common fixes are:

- Reinstall DGL from a wheel matching PyTorch 2.12 + CUDA 12.6 (or CPU).
- Downgrade PyTorch to a version for which a DGL wheel exists.
- Do not import DGL at all.

### 3. Does the project actually need DGL?

**No.** The project does not import or use DGL directly:

```bash
$ grep -R "import dgl\|from dgl\|\bdgl\b" --include='*.py' .
# no matches
```

The ALIGNN backbone uses `alignn.models.alignn_atomwise_pure`, which is a **pure-PyTorch** implementation that does **not** require DGL:

```python
from alignn.models.alignn_atomwise_pure import (
    ALIGNNAtomWisePure,
    ALIGNNAtomWisePureConfig,
)
```

Verification:

```bash
$ python -c "from alignn.models.alignn_atomwise_pure import ALIGNNAtomWisePure, ALIGNNAtomWisePureConfig; print('OK')"
OK

$ python -c "import backbones; print('backbones import OK')"
backbones import OK

$ python -m pytest tests/test_backbones.py -v
12 passed, 23 warnings in 7.19s
```

So the code runs fine even though `import dgl` fails. The broken DGL package is an unused transitive dependency of `alignn` (or another installed package) and does not block training or testing.

### 4. Recommendation for dependency documentation

There is currently **no** `pyproject.toml`, `requirements.txt`, `setup.py`, or `setup.cfg` in the project. Create one of the following:

**Option A — `requirements.txt` (simplest for a research repo):**

```text
# Core
torch>=2.0,<2.13
numpy
pytest

# Materials / data
pymatgen
jarvis-tools

# Graph backbones (PyG-based or pure-PyTorch; DGL is NOT required)
torch-geometric
egnn-pytorch
alignn>=2026.5.20
matgl
```

**Option B — `pyproject.toml` (modern, ICLR-friendly):**

```toml
[project]
name = "fr-phytca"
version = "0.1.0"
description = "Fidelity-Residual Physics-Structured Tensor Component Adaptation"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0,<2.13",
    "numpy",
    "pytest",
    "pymatgen",
    "jarvis-tools",
    "torch-geometric",
    "egnn-pytorch",
    "alignn>=2026.5.20",
    "matgl",
]

[project.optional-dependencies]
dgl = ["dgl"]  # not required; only for users who want the legacy DGL-based ALIGNN path
```

**README update:** Add a note such as:

> This codebase uses `alignn.models.alignn_atomwise_pure`, a pure-PyTorch ALIGNN implementation. **DGL is not required.** If you have DGL installed and see a `graphbolt` DLL error on PyTorch 2.12, you can safely ignore it or uninstall DGL; training and tests do not depend on it.

### 5. No installations performed

As requested, I did **not** install, uninstall, or upgrade any packages.

---

## Summary checklist

- [x] Remote reachable; `origin/main` at `209ebcb`.
- [x] Local `main` is 1 commit ahead; **do not push yet** until you decide on the new repo.
- [x] No sensitive files; large artifacts are correctly `.gitignore`d.
- [x] Commit history is clean enough for ICLR submission.
- [x] Old repo can be deleted **after** preserving the local-only commit.
- [x] `alignn` imports successfully; DGL import fails due to PyTorch 2.12 graphbolt DLL mismatch.
- [x] Project runs without DGL; ALIGNN backbone uses pure-PyTorch path.
- [x] No `requirements.txt`/`pyproject.toml` exists; recommendations provided above.
