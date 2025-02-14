# Contributor's Guide

(contributing:getting-started)=
## Getting Started

These contributing guidelines should be read by software developers wishing to contribute code or
documentation changes into WOMBAT, or to push changes upstream to the main WISDEM/WOMBAT repository.

1. Create a fork of WOMBAT on GitHub
2. Clone your fork of the repository

    ```bash
    git clone -b develop https://github.com/<your-GitHub-username>/WOMBAT.git
    ```

3. Move into the WOMBAT source code directory

    ```bash
    cd WOMBAT/
    ```

4. Install WOMBAT in editable mode with the appropriate developer tools

   - ``".[dev]"`` is for the linting, autoformatting, testing, and code checking tools
   - ``".[docs]"`` is for the documentation building tools. Ideally, developers should also be
     contributing to the documentation, and therefore checking that the documentation builds locally.

    ```bash
    pip install -e ".[dev,docs]"
    ```

5. Turn on the automated linting and code checking tools. Pre-commit runs at the commit level, and
   will only check files that have been modified and staged (e.g., `git add <your-changed-file>`).

   ```bash
   pre-commit install
   ```

## Keeping your fork in sync with WISDEM/WOMBAT

The "main" WOMBAT repository is regularly updated with ongoing research at NREL and beyond. After
creating and cloning your fork from the previous section, you might be wondering how to keep it
up to date with the latest improvements.

Please note that the below process may introduce merge conflicts with your work, and this does not
provide guidance about how to deal with those conflicts. Here is a good resource for working on
[merge conflicts](https://www.atlassian.com/git/tutorials/using-branches/merge-conflicts) that will
inevitably arise in development work.

1. Ensure you're in the WOMBAT folder. This may look different depending on your operating system.

   ```bash
   cd /your/path/to/WOMBAT/
   ```

2. If you haven't already, add WISDEM/WOMBAT as the "upstream" location (or whichever naming
   convention you prefer).

   ```bash
   git remote add upstream https://github.com/WISDEM/WOMBAT.git
   ```

   To find the name you've given WISDEM/WOMBAT again, you can simply run the following to display
   all the remote sources you're tracking.

   ```bash
   git remote -v
   ```

3. Fetch all the remote changes

   ```bash
   git fetch --all
   ```

4. Sync the upstream (WISDEM) changes

   ```bash
   # If there was a new release this will need to be updated
   git checkout main
   git pull upstream main

   # Most common branch to bring up to speed
   git checkout develop
   git pull upstream develop
   ```

5. Bring your feature branch up to date with the latest changes, assuming you started from the
   develop branch.

   ```bash
   git checkout feature/your_contribution
   git merge develop
   ```

## Issue Tracking

New feature requests, changes, enhancements, non-methodology features, and bug reports can be filed
as new issues in the [Github.com issue tracker](https://github.com/WISDEM/WOMBAT/issues) at any time.
Please be sure to fully describe the issue.

For other issues, please email rob.hammond@nrel.gov.

### Issue Submission Checklist

1. Does the issue already exist?
   Yes: If you find your issue already exists, make relevant comments and add your
   [reaction](https://github.com/blog/2119-add-reactions-to-pull-requests-issues-and-comments).
   Use a reaction in place of a "+1" comment:

   - 👍 - upvote
   - 👎 - downvote

2. Is this an individual bug report or feature request?
3. Can the bug be easily reproduced?
   1. Be sure to include enough details about your setup and the issue you've encountered
   2. Simplify as much of the code as possible to better isolate the problem
4. Will someone else understand the issue or change requested given the information provided?

## Repository

The WOMBAT repository is hosted on Github, and located here: http://github.com/WISDEM/WOMBAT

This repository is organized using a modified git-flow system. Branches are organized as follows:

- main: Stable release version. Must have good test coverage and may not have all the newest features.
- develop: Development branch which contains the newest features. Tests must pass, but code may be
  unstable.
- feature/xxx: Feature ranch from develop, should reference a GitHub issue number.
- fix/xxx: Bug fix branch from develop, should reference a GitHub issue number. Can be based off
  main if this is a necessary patch.

To work on a feature, please fork WOMBAT first and then create a feature branch in your own fork.
Work out of this feature branch before submitting a pull request.

Be sure to periodically synchronize the upstream develop branch into your feature branch to avoid
conflicts in the pull request.

When your branch is ready, make a pull request to WISDEM/WOMBAT through the
[GitHub web interface](https://github.com/WISDEM/WOMBATpulls).

## Coding Style

This code uses a ``pre-commit`` workflow where code styling and linting is taken care of when a user
commits their code. Specifically, this code utilizes ``black`` for automatic formatting (line
length, quotation usage, hanging lines, etc.), ``isort`` for automatic import sorting, ``mypy``
for typing, and ``ruff`` for linting.

To activate the ``pre-commit`` workflow, the user must install the developer version as outlined in
the [getting started section](contributing:getting-started), and run the following line:

```bash
pre-commit install
```

## Documentation

Documentation is written primarily using Markdown, with some components written in
ReStructured Text, and is located in the `WOMBAT/docs/` directory. Additionally, all method and class
documentation is written as NumPy-style docstrings in the code itself, with some aspects documented
inline as needed.

If the `docs` extras haven't already been installed, be sure to do so before you attempt to build
the documentation site.

```bash
# Navigate to the top level directory of the repository
cd WOMBAT/

# Install the additional dependencies
pip install -e ".[docs]"
```

Now, run the build command as follows.

```bash
# Build the docs
jupyter-book build docs
```

## Testing

All code should be paired with a corresponding unit, regression, or integration test written with
the pytest framework.

To run the tests you can use any of the following commands, depending on your needs.

1. All the tests and check for test coverage:

   ```bash
   pytest --cov=WOMBAT
   ```

2. All the tests:

   ```bash
   pytest
   ```

(contributing:pull-request)=
## Pull Request

Pull requests must be made for all changes. Most pull requests should be made against the develop
branch unless patching a bug that needs to be addressed immediately, and only core developers should
make pull requests to the main branch.

All pull requests, regardless of the base branch, must include updated documentation and pass all
tests. In addition, code coverage should not be significantly negatively affected.

### Scope

Encapsulate the changes of one issue, or multiple if they are highly related. Three small pull
requests is greatly preferred over one large pull request. Not only will the review process be
shorter, but the review will be more focused and of higher quality, benefitting the author and code
base. Be sure to write a complete description of these changes in the pull request body.

### Tests

All tests must pass. Pull requests will be rejected or have changes requested if tests do not pass,
or cannot pass with changes. Tests are automatically run through Github Actions for any pull request
or push to the main or develop branches, but should also be run locally before submission.

#### Test Coverage

The testing framework described below will generate a coverage report from the tests run through
GitHub Actions. Please ensure that your work is fully covered by running them locally with the
coverage report enabled.

### Documentation

Include any relevant changes to inline documentation, docstrings, and any of the documentation files
located in `WOMBAT/docs/`. Pull requests will not be accepted until these changes are complete.

Be sure to build the documentation and run the examples in the CLI locally prior to submission.

### Changelog

All changes must be documented appropriately in CHANGELOG.md in the [Unreleased] section.

## Release Process

This section is a reference for WOMBAT's maintainers to keep processes largely consistent
over time, regardless of who the core developers are.

1. Rerun tests
2. Rebuild the documentation locally
   1. Recreate the example notebooks

     ```bash
     jupytext --to notebook docs/examples/how_to.md docs/examples/metrics_demonstration.md docs/examples/strategy_demonstration.md
     ```

   2. Move the notebooks to the examples folder

     ```bash
     mv docs/examples/*.ipynb examples
     ```

3. Bump version number and metadata in `WOMBAT/__init__.py`
4. Bump version numbers of any dependencies in `setup.py`. Be sure to separate to keep dependencies
   separated by what they are required for (see the `project.optional-dependencies` section of
   `pyproject.toml`)
5. Update the changelog at `WOMBAT/CHANGELOG.md`, changing the "UNRELEASED" section to the new
   version and the release date (e.g. "[2.3 - 2022-01-18]").
6. Make a pull request into develop with these updates, and be sure to follow the guide in
   [Pull Requests](contributing:pull-request).

7. Merge develop into main through the git command line

  ```bash
   git checkout main
   git merge develop
   git push
   ```

- Tag the new release version:

  ```bash
   git tag -a v1.2.3 -m "Tag message for v1.2.3"
   git push origin v1.2.3
   ```

   The above process will trigger the `.github/workflows/python-publish-test.yml` GitHub Action
   that builds the package and pushes it to Test PyPI. If this is successful, you can move to the
   next step, otherwise the errors from the action should be addressed, and the tag should be
   deleted, then recreated after the fix is published.

- Deploying a Package to PyPi
  - The repository is equipped with a GitHub Action to build and publish new versions to PyPI. A
    maintainer can invoke this workflow by creating a new release on GitHub that corresponds to the
    created tag in the previous step.
  - The action is defined in `.github/workflows/python-publish.yml`.
