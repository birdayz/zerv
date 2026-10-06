# Repository publication privacy

Scope: publishable Git content (all branches and tags), commit/tag messages and
identity metadata, plus tracked and non-ignored working files. Ignored model
artifacts, tool caches, local credentials and recovery backups are not publication
inputs; do not upload them. This is a repository hygiene policy, not an inference
engine feature or a guarantee that a heuristic scanner detects every secret.

## Data allowed in the repository

- Public source citations and author attribution, dependency/model hashes,
  synthetic correctness fixtures and explicitly public benchmark workloads.
- Hardware model, capacity, driver/compiler versions and performance measurements
  needed to reproduce experiments, but not machine/account identifiers.
- Loopback addresses and documented synthetic/example addresses.

## Data excluded from publication

- Credentials, private keys, authenticated URLs, session cookies, access tokens
  and local environment/configuration dumps.
- Personal account paths, private email addresses, hostnames, hardware serial
  numbers, device/filesystem identifiers and private-network configuration.
- Private conversations, unrelated personal files and recovery/source locations.
  Keep only the technical conclusions needed by the project.

Replace local account roots in evidence with `/home/USER`, retaining the relative
path and measured values. Redaction is not a new benchmark run. If an evidence
file is redacted, its original recorded digest identifies the original private
bytes, not the sanitized file. Do not silently recompute historical measurements
or claim their byte-for-byte provenance survived redaction.

## Verification and remediation

1. Inventory tracked and non-ignored files; check filenames and binary content as
   well as text. Review identity-bearing hardware captures and prompt transcripts.
2. Run a pinned independent credential scanner over an isolated copy of this
   inventory and over all reachable Git history, without suppressing repository
   paths. Classify every finding; synthetic examples need evidence, not assumptions.
3. Inspect remote refs and Git author/committer/tagger metadata separately; secret
   scanners do not establish that personal identifiers are absent.
4. Apply only reviewed redactions. Preserve unrelated worktree/index changes and
   benchmark values. Re-scan the worktree and rewritten history; verify ref updates.
5. History rewriting and force-pushing require explicit user authorization. Use
   leases pinned to inspected remote tips; do not overwrite concurrent updates.
   Existing clones, hosting-provider caches and hidden pull-request refs can retain
   prior bytes even after a successful force-push. Revoke/rotate any real leaked
   credential; rewriting history does not invalidate it.

Use owner-only local reports and backups outside publication inputs; never paste
secret matches into the audit report. `.gitignore` is defense in depth, not a
scanner, and does not protect files already tracked or forcibly added. Before
publishing fresh experiment output, repeat the scan: subprocess logs and absolute
paths can reintroduce local identity information.

Acceptance is an explicitly scoped report of actual scans, reviewed findings,
redactions and remaining limitations, not an unqualified “no private data” claim.
