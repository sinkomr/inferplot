# Releasing inferplot

This is the owner's checklist for publishing a version to PyPI, written for a first
release. Everything up to the tag is prepared in the repository; the steps below
are the ones only the owner can take, because each one publishes or grants the
right to publish. Nothing is published unless you push the tag (step 3, which
sends the files to TestPyPI, the practice site) and then approve the waiting job
(step 4, which sends them to PyPI).

How it works, in one paragraph: pushing a tag such as `v0.1.0` starts
`.github/workflows/release.yml`. It builds the package, checks it (the tag equals
the version, the wheel holds only the package, the wheel installs and runs the
README's quick start on Python 3.11 and 3.15, the test suite passes from the sdist),
uploads it to TestPyPI, and then stops and waits. PyPI gets the files only after
you approve the waiting job on GitHub. No password or token is stored anywhere:
PyPI trusts this one workflow file in this one repository ("trusted publishing").

The approval exists only if the GitHub environment `pypi` has you as a required
reviewer (step 0.4). GitHub would otherwise create that environment by itself,
unprotected, and nothing would wait. The workflow therefore checks this before it
builds anything and stops with an error if no reviewer is required; do not remove
that check, and do step 0.4 before you register the publisher on PyPI (step 0.2).

Values used below (they must match exactly; PyPI compares them as text):

| Setting | Value |
| --- | --- |
| PyPI project name | `inferplot` |
| Owner (GitHub user) | `sinkomr` |
| Repository name | `inferplot` |
| Workflow file name | `release.yml` |
| Environment name on PyPI | `pypi` |
| Environment name on TestPyPI | `testpypi` |

## 0. Before the first release (once, about 30 minutes)

You need two accounts: one on [pypi.org](https://pypi.org/account/register/) and a
separate one on [test.pypi.org](https://test.pypi.org/account/register/) (TestPyPI is
a practice copy of PyPI with its own accounts). Turn on two-factor authentication on
both; PyPI requires it for publishing.

1. **Check that the name is free.** Open <https://pypi.org/project/inferplot/> and
   <https://test.pypi.org/project/inferplot/>. "Not Found" means it is free. If
   either exists and is not yours, stop: the name in `pyproject.toml` has to change
   first.
2. **Register the trusted publisher on PyPI.** Signed in to pypi.org: your account
   menu > *Your projects* > *Publishing* > *Add a new pending publisher* > GitHub.
   Enter the values from the table: project name `inferplot`, owner `sinkomr`,
   repository `inferplot`, workflow `release.yml`, environment `pypi`. A pending
   publisher reserves nothing until the first upload; the project is created by
   that upload.
3. **Do the same on TestPyPI** (test.pypi.org > *Publishing*), with environment
   `testpypi`.
4. **Create the two GitHub environments.** On GitHub: the repository > *Settings* >
   *Environments* > *New environment*.
   - `pypi`: tick *Required reviewers* and add yourself. Under *Deployment branches
     and tags* choose *Selected branches and tags* and add the tag rule `v*`. Also
     tick *Prevent self-review* only if someone else will approve; for a one-person
     project leave it off, or you cannot approve your own release.
   - `testpypi`: same tag rule `v*`, plus the branch `main` (so a practice run from
     main can reach TestPyPI). Required reviewers are optional here.
5. **Protect the tags (recommended).** *Settings* > *Rules* > *Rulesets* > *New tag
   ruleset*: target `v*`, restrict creation, updates and deletion to yourself. Then
   nobody else can start a release by pushing a tag.

The workflow's actions are pinned by commit (the long hexadecimal strings after
`@` in `release.yml`), so a changed tag cannot change what a release runs. The pins
are the versions `ci.yml` and `oracle.yml` use by tag, checked against GitHub on
2026-11-10. To check them again, or after raising one:

```sh
gh api repos/actions/checkout/commits/v7.0.1 --jq .sha
gh api repos/actions/upload-artifact/commits/v7.0.2 --jq .sha
gh api repos/actions/download-artifact/commits/v8.0.2 --jq .sha
gh api repos/astral-sh/setup-uv/commits/v10.3.3 --jq .sha
```

Each must print the SHA used in `release.yml` (`3d3c42e5...`, `cf430e03...`,
`9000827c...`, `68559e99...`).

## 1. Prepare the release commit (on a branch, merged to main)

1. Set the version in `src/inferplot/__init__.py`: `__version__ = "0.1.0"` (it is
   `0.1.0.dev0` during development; this line is the only place the version lives).
2. In `CHANGELOG.md`, change `## [0.1.0] - Unreleased` to `## [0.1.0] - YYYY-MM-DD`
   with the day's date.
3. In `README.md`, replace the "not released" status and the install-from-a-clone
   instructions with `pip install "inferplot[pandas]"`; in
   `skills/inferplot/SKILL.md` change the sentence that says it is not on PyPI; in
   `llms.txt` the sentence "Not yet on PyPI: install from the repository." The
   release check refuses a final version whose description still says "not
   released" or "not on PyPI".
4. Merge to main and wait until every CI job is green, the lowest-dependency job
   included.

`tests/test_packaging.py` fails if the version and the CHANGELOG disagree, so a
mismatch shows up in CI before any tag exists.

## 2. Optional: a practice run on TestPyPI

A practice run uploads a pre-release to TestPyPI only. TestPyPI, like PyPI, never
accepts the same version twice, so practise with a pre-release number:

Do it before the release commit of step 1, or set the version back afterwards:
the tag of step 3 must point at a commit whose version is exactly `0.1.0`.

1. Set `__version__ = "0.1.0rc1"` on main (the CHANGELOG entry stays `0.1.0`).
2. GitHub > *Actions* > *Release* > *Run workflow*, branch `main`, target
   `testpypi` > *Run workflow*.
3. When it is green, install it in a fresh environment:

   ```sh
   uv venv /tmp/try-inferplot
   uv pip install --python /tmp/try-inferplot \
     --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ \
     --index-strategy unsafe-best-match "inferplot[pandas]==0.1.0rc1"
   /tmp/try-inferplot/bin/python -c "import inferplot; print(inferplot.__version__)"
   ```

   (The extra index supplies numpy, scipy and the other dependencies, which
   TestPyPI does not carry.) A second practice run needs `0.1.0rc2`, and so on.
4. When you are done practising, set `__version__` to `"0.1.0"` (step 1.1) before
   you tag. If main still says `0.1.0rc1`, the workflow refuses the tag `v0.1.0`
   and publishes nothing.

The target `build-only` (the default) builds and checks without publishing; it is
safe to run at any time.

## 3. Tag the release

First check, on GitHub under *Settings* > *Environments* > `pypi`, that *Required
reviewers* is ticked with your name (step 0.4). Then, on an up-to-date main whose
last commit is the release commit (version `0.1.0`), from a terminal:

```sh
git switch main && git pull
git tag -a v0.1.0 -m "inferplot 0.1.0"
git push origin v0.1.0
```

The tag must be `v` followed by exactly the version in `__init__.py`; the workflow
refuses anything else before it publishes.

## 4. Watch, then approve PyPI

GitHub > *Actions* > *Release*, the run named after the tag. It takes about 10 to
15 minutes:

1. *Build and check the packages* (which first checks that the `pypi` environment
   requires a reviewer), *Install the wheel* (3.11 and 3.15; it runs the README's
   quick start and draws one figure with each of the five functions) and *Test
   from the sdist* run first. If any fails, nothing has been published: fix the
   problem on main, delete the tag (`git push origin :refs/tags/v0.1.0`,
   `git tag -d v0.1.0`) and tag again.
2. *Publish to TestPyPI* uploads. Check <https://test.pypi.org/project/inferplot/>
   and, if you like, install from it as in step 2.
3. *Publish to PyPI* shows *Waiting*. GitHub emails you. Open the run, *Review
   deployments*, tick `pypi`, *Approve and deploy*. This is the moment of
   publication; after it the version is public and cannot be replaced.

## 5. Check what was published

- <https://pypi.org/project/inferplot/> shows 0.1.0, the README, and two files: a
  wheel (`inferplot-0.1.0-py3-none-any.whl`, about 0.5 MB) and an sdist
  (`inferplot-0.1.0.tar.gz`, about 6 MB, which carries the tests and their data).
- In a fresh environment: `pip install "inferplot[pandas]"`, then run the README's
  quick start; it prints the three lines the README shows.
- Optionally create a GitHub release from the tag (*Releases* > *Draft a new
  release*, choose `v0.1.0`, paste the CHANGELOG entry). The workflow does not do
  this.

## If TestPyPI succeeded but PyPI must wait

Do nothing: the PyPI job waits for approval for up to 30 days and then expires.
To stop it at once, open the run and *Reject* (or *Cancel workflow*). TestPyPI
keeps the version, which does no harm. When ready, either approve the same run (if
still waiting) or fix main, bump to the next version (TestPyPI and PyPI never take
a version twice) and tag again. Re-running only the failed or rejected PyPI job
from the run page works too, as long as the tag still points at the same commit.

## If a published version is wrong: yank, then release a fix

A published version cannot be changed or re-uploaded. If 0.1.0 is broken:

1. Fix it on main, set `0.1.1`, add a CHANGELOG entry, tag `v0.1.1` as above.
2. Yank 0.1.0: pypi.org > *Your projects* > inferplot > *Manage* > *Releases* >
   0.1.0 > *Options* > *Yank*, with a one-line reason. A yanked version stays
   downloadable for anyone who pins it exactly (`inferplot==0.1.0`), but `pip`
   no longer picks it otherwise. Deleting a release is possible but leaves the
   version number unusable forever and breaks anyone pinned to it; yank instead.

## What the owner is asked to approve, in short

| Step | Who | Publishes? |
| --- | --- | --- |
| 0. Accounts, pending publishers, environments, tag ruleset | owner | no |
| 1. Release commit merged to main | lead, owner reviews | no |
| 2. Practice run (target `testpypi`) | owner starts it | TestPyPI only |
| 3. Push the tag | owner | TestPyPI after the checks |
| 4. Approve the `pypi` environment | owner | PyPI |
