# Releasing mcts-agent

Package versions come from Git tags through `setuptools-scm`. Create and push a
version tag to run the release workflow:

```bash
git tag v0.4.0
git push origin v0.4.0
```

The workflow runs the test suite, verifies that the package version matches the
tag, builds a wheel and source distribution, creates a GitHub Release with
generated notes and distribution files, then publishes those files to PyPI.
The release workflow is `.github/workflows/release.yml`.

## One-time PyPI setup

Configure a PyPI Trusted Publisher for:

- Owner: `lhemerly`
- Repository: `mcts-agent`
- Workflow filename: `release.yml`
- GitHub environment: `pypi`

The PyPI job uses OIDC and does not need a stored PyPI API token. The `pypi`
environment can also be configured in GitHub repository settings for deployment
protection rules. Set up the trusted publisher before pushing the first release
tag; otherwise the workflow will fail at the publish step.

Use a new, valid PEP 440 version for each tag. For example, `v0.4.0` produces
package version `0.4.0`. Do not reuse a version already uploaded to PyPI.
