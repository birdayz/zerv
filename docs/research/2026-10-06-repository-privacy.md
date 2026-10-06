# Repository privacy audit — 2026-10-06

Scope: tracked and non-ignored working files, all advertised branches, all reachable
history and four registered worktree HEADs. User explicitly authorized history
rewriting and force-pushing. This maintenance task does not advance the active
inference-performance building block. [Publication policy](../specs/repository-privacy.md).

## Findings and remediation

No confirmed account-authentication credential or private key was found. This is
an observed, scoped audit result, not proof that a scanner recognizes every secret.
The following identifying data was found and redacted:

- Local home/account paths (including build-cache account components) and filesystem
  owner/group output in benchmark transcripts, logs and command manifests.
- Host identity in JSON `node`/`nodename` fields, `uname` arrays and kernel-log
  prefixes; the same ordinary word in prose and public tokenizer data is retained.
- A GPU device UUID and four account-specific skill-sync IDs. Driver/build UUIDs,
  public model/source hashes and synthetic request IDs are not personal identities.
- Recovered-home and shell-history details in the old deployment note. Relevant
  Ollama configuration, public artifact source and uncertainty remain documented.
- A bot-management cookie and opaque response/trace identifiers in an untracked
  download-header capture. The expired cookie was not proven account authentication;
  it was removed regardless. No genuine account credential requiring rotation was
  identified by this audit.
- Personal author/committer metadata, replaced by `zerv contributors
  <contributors@example.invalid>`. This is a reserved, non-deliverable address,
  not another person's identity.
- The architecture diagram's ancillary C2PA provenance chunk (23,654 bytes of
  payload). The image-data chunks are preserved byte-for-byte.

Public packager attribution, public benchmark-result IDs, Docker's default bridge
address, loopback addresses and generated InferenceX random-token text were reviewed
and retained. Apparent IP addresses in one log were dotted timing fields, not LAN
addresses. The bruh demonstrations use synthetic hello/calculator tasks and
repository-summary prompts; their personal execution metadata, not the task data,
was redacted.

Local credentials, dotenv files, key files and browser HAR exports now have ignore
rules. Ignore rules are not a security boundary for already tracked or forced files.
New experiment output must still be reviewed before publication.

## Independent credential scans

Pinned Gitleaks **v8.30.1**, official Linux x64 release:

- [Release](https://github.com/gitleaks/gitleaks/releases/tag/v8.30.1).
- Archive SHA-256:
  `551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb`.
- Executable SHA-256:
  `88f91962aa2f93ac6ab281d553b9e125f5197bbbce38f9f2437f7299c32e5509`.
- [Inspected source](https://github.com/gitleaks/gitleaks/tree/v8.30.1), retained
  under gitignored `third_party/gitleaks/v8.30.1/source/`; source-archive SHA-256
  `e90fb266d75837e75894c778bf594ab8e2787f12dce5a62651f21b893eaf9abb`.
- Release archive/checksums retained beside the source. Verified against the official
  checksum list before execution; this is integrity checking, not an independent
  signature audit. No system package installation.

Initial inventory: **13,570 files / 1,090,814,711 bytes**; **60 commits**, main and
bazel plus their remote-tracking refs. Both detached worktree HEADs are contained in
those same 60 commits. `git ls-remote --refs origin` advertised no additional refs.
Git messages and author/committer metadata were inspected separately from diffs.

| Initial scanner result | History | Working snapshot |
| --- | ---: | ---: |
| Artifact/source SHA-256 false positives | 942 | 965 |
| Public InferenceX result identifiers | 41 | 41 |
| Boolean benchmark setting in prose | 21 | 21 |
| Total `generic-api-key` findings | 1,004 | 1,027 |
| Unresolved findings | **0** | **0** |

Every match was classified against its original source line. The scanner returned
exit 1 for these findings; it did **not** return a clean zero-finding result for the
main scans. No broad repository allowlist was added to hide benchmark directories.

The initial directory scan's 715,724,152-byte total was reconciled exactly:
1,090,814,711 inventory bytes minus 387,574,146 built-in `.bin`/PNG exclusions plus
12,483,587 gzip expansion. Those default exclusions were covered independently:

- 209 current `.bin` copies collapse to nine distinct payloads. Those plus 281
  historical SPIR-V objects and PNG provenance were staged as hash-named `.txt`
  files with a short text prefix, retaining every original byte. **18,237,655
  scanned bytes, zero findings.** This bypassed extension/magic skipping.
- All four gzip streams were explicitly decompressed: **19,258,878 bytes, zero
  findings**; benign stored basenames, no embedded private account identifiers.
- Commit messages: **27,911 bytes, zero findings**.
- Supplemental raw-byte checks covered credential assignments/provider tokens,
  private-key markers, authenticated URLs, sensitive filenames and known private
  identifiers. Regex checks alone were not treated as clearance.

Reproducible scan shape (use owner-only reports, a fresh isolated working snapshot,
Gitleaks defaults explicitly enabled, and an empty ignore file):

```sh
git ls-files -z --cached --others --exclude-standard
# Copy exactly that inventory into a fresh SNAPSHOT, not build/model/cache directories.
GITLEAKS=third_party/gitleaks/v8.30.1/gitleaks
"$GITLEAKS" git . --log-opts=--all --redact=100 --ignore-gitleaks-allow \
  --config DEFAULT_CONFIG --gitleaks-ignore-path EMPTY_IGNORE \
  --max-archive-depth 3 --report-format json --report-path PRIVATE_HISTORY_REPORT
"$GITLEAKS" dir SNAPSHOT --redact=100 --ignore-gitleaks-allow \
  --config DEFAULT_CONFIG --gitleaks-ignore-path EMPTY_IGNORE \
  --max-archive-depth 3 --report-format json --report-path PRIVATE_TREE_REPORT
```

`DEFAULT_CONFIG` contains `[extend]` followed by `useDefault = true`. Gitleaks's own
rule allowlists/entropy thresholds still apply. Independently inspect metadata,
archives and default binary exclusions as above. Private reports/replacement values
and recovery copies are deliberately not publication artifacts.

## Rewrite and verification

History rewriting used **git-filter-repo 2.47.0** (tool version `a40bce548d2c`),
installed with `uv` under ignored `third_party/git-filter-repo/2.47.0` and run with
the pinned `tools/py` interpreter. Source: [upstream release](https://github.com/newren/git-filter-repo/releases/tag/v2.47.0).
An isolated mirror was rewritten using reviewed blob/message callbacks and neutral
name/email callbacks before any live ref update. The installed `git_filter_repo.py`
SHA-256 is `67447413e273fc76809289111748870b6f6072f08b17efe94863a92d810b7d94`.
An owner-only recovery copy of Git history and all four working inventories was
created outside the repository.

All **60 rewritten commits** were checked for parent mapping and neutral metadata.
All **1,787 distinct changed old/new blob pairs** were compared to the reviewed
redaction transform; **1,961 historical paths** changed. Changes outside `docs/`
are limited to command-path metadata in two historical model-oracle JSON fixtures.
The public tokenizer binary stays byte-identical. Redactions preserve executable
code, numerical fixtures, measurement values and file modes. Full working-inventory
comparison verified 13,571 / 9,957 / 9,804 / 9,957 files across the four worktrees
against the backed-up bytes plus only the reviewed redactions and explicit privacy
policy/index edits. The current tree had 2,135 redacted evidence files, all under
`docs/`. The three other worktrees are clean against their rewritten bases;
pre-existing dirty/untracked work in the current tree remains uncommitted.

A pre-publication check caught an overbroad bare-hostname substitution: the host
name is also an ordinary word in prose and the public tokenizer vocabulary. That
candidate was rejected, affected working files were restored from the private
backup with context-only redactions, and the mirror was rebuilt from the original.
No branch ref or remote had been changed at that point. A read-only evidence file
also rejected an in-place write; retry used atomic replacement preserving its mode.
These were remediation failures, not successful first-pass results.

`bazel test //...`: 87/87 cached results. A subsequent
`bazel test --nocache_test_results //...` **executed all 87 tests and passed**.
No production GPU/kernel change was made, so no new GPU performance claim or
hardware benchmark is associated with this maintenance work.

### Final post-redaction gates and publication

Independent repeat scans completed on the isolated corrected mirror and a fresh
sanitized main-worktree snapshot:

- Mirror: **60 commits, 8,121 unique blobs / 360,527,807 raw bytes**; four gzip
  expansions also scanned. Original reviewed private values and contextual
  hostname matches: **0**. Neutral author/committer metadata: **60/60**.
- Working snapshot: **13,572 files / 1,090,758,224 raw bytes**, no missing or
  concurrently changed snapshot inputs. Original reviewed private values and
  contextual hostname matches: **0**, including decompressed archives.
- Primary scanner counts remain **1,004 / 1,027** (mirror / working snapshot),
  all source-verified members of the same three false-positive classes above;
  **zero unresolved findings**. Independent binary/archive/message checks return
  exit 0 with no findings.
- Independent targeted review of the 17 originally identifying artifacts found no
  remaining private identifiers. PNG now contains only IHDR/IDAT/IEND chunks.
- Ignore-rule checks pass: local secrets excluded, `.env.example` and benchmark
  logs retained. Fresh CPU/Python/format/generated-file test gate: **87/87 passed**.

The rewritten **main and bazel branches were published atomically** using
`git push --atomic --force-with-lease=refs/heads/main:OLD_MAIN
--force-with-lease=refs/heads/bazel:OLD_BAZEL origin
refs/heads/main:refs/heads/main refs/heads/bazel:refs/heads/bazel`.
Both inspected old-tip leases matched; an independent `git ls-remote --refs origin`
confirmed both new tips. No unrelated dirty work was committed or pushed. The
privacy policy, ignore rules, index links and this report are a separate follow-up
commit, not a publication of the ongoing performance work.

Local temporary rewrite refs were removed, stale fetch metadata cleared, original
head pointers remapped, all worktree reflogs expired and old objects pruned after
creating the external recovery backup. All 60 original commit objects are absent
from the live local object store; `git fsck --full --no-reflogs` passes. The repository's
local Git author/committer defaults now use the neutral identity; global settings
are untouched. Private audit reports and recovery material are retained outside
publication inputs and must not be uploaded.

## Evidence and retention limitations

Recorded historical digests refer to original experiment bytes; redacted logs,
manifests and transcripts may no longer match those digests. Original source/build
hashes and measurements have not been fabricated or silently regenerated. A
transcript with redacted metadata is not an exact-byte replay input. No inference
or benchmark was rerun to imply otherwise.

Force-pushing removes old history from advertised branches, not from existing
clones, forks, hosting-provider caches or hidden pull-request refs. Fresh clones
or deliberate reconciliation are required; merging an old branch can reintroduce
the removed data. Provider-side cached-object purging may require repository-owner
support. This audit does not claim such copies were erased.

Ignored local models, tools, build caches and private recovery material are not
publishable repository content and are not globally sanitized. Local Git/OS paths
and the user's global Git identity are not publication artifacts. No system-wide
identity setting is changed. Ongoing benchmark tools can emit local paths again;
repeat the publication checks for new output.
