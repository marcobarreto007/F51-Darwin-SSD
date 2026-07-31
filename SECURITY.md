# Security policy

## Supported scope

Security maintenance covers the tracked local single-root source on the current
default branch, CPython 3.12, and the supported entrypoints declared in
`governance/audit/policy/operational-surface.json`. Files below `governance/archive/`, cloud-era
workflows, historical checkpoints and operator-modified environments are not a
supported security surface.

## Reporting

Report suspected vulnerabilities privately to Marco Barreto / F51 Labs through
an established private operator channel. Do not open a public issue containing
credentials, exploit details, private corpus content, checkpoint data or local
machine identifiers. Include the affected commit, path, reproduction steps,
impact, and a minimal redacted proof.

## Secrets

No credential belongs in Git, logs, manifests, tests or examples. If a real
secret is detected, stop publication, rotate or revoke it first, preserve a
redacted incident record, and make any history-rewrite decision separately.
Never print a discovered value into an audit report.

## Dependency findings

Unexcepted High or Critical vulnerabilities block release. An exception must
name the advisory, owner, reason, mitigation and an unexpired UTC deadline in
`governance/audit/policy/dependencies.json`.
