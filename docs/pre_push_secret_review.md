# Pre-push credential review

This review checks the local archive before its first GitHub push. It does not
upload repository contents, call a credential provider, test a token online or
change account credentials.

## Scope and findings

- The IBM/Qiskit account file is outside the repository, in the user's home
  configuration directory. It is not tracked by Git.
- One locally configured credential value was compared in memory against all
  2,087 unique file blobs reachable through the 26 local commits and against the
  tracked working tree. Raw, URL-encoded, base64 and hexadecimal forms were
  checked. ZIP-based NPZ/PT members were decompressed for exact matching.
  **No matches were found.** No secret value or secret fingerprint is stored here.
- The known credential was also checked against 434 reachable commit/tag/tree
  objects and local Git ref names. No matches were found in that metadata.
- Official Gitleaks v8.30.1 was downloaded to temporary storage and checked
  against both its published checksum file and release-asset SHA-256. The tool
  is not a project dependency and is not included in the repository.
- Gitleaks scanned all reachable commit patches and an exported tracked-file
  tree, with archive/decode support, bundled default rules, full output redaction,
  no baseline suppression, and no repository/inline allowlist suppression.
- Each scan reported 101 generic-API-key alerts. **All 101 in each scan were
  independently verified as source-file SHA-256 records**, by hashing the named
  source file in the relevant commit, working tree or historical version. No
  unresolved alerts remained. Raw result manifests were not edited to silence
  the scanner, and no broad checksum allowlist was added.
- No credential-like account/key files are tracked. There are 24 previously
  tracked checkpoint/config files under an existing ignore rule; ignoring a
  tracked file does not untrack it. These legitimate research files were included
  in the scans and were retained.

The `.gitignore` now covers dotenv files, local IBM/Qiskit/cloud/SSH/Codex account
directories, common credential filenames and private-key containers. Template
filenames `.env.example` and `.env.template` remain permitted; they must contain
placeholders only. Ignore rules do not protect a secret pasted into an already
tracked source file or remove a credential from history.

## Interpretation

No IBM credential or other actual secret was detected in the reviewed pushable
content. This is a scoped scan result, not a guarantee about all unknown or
future secrets. Exact matching covers locally readable known values, not unknown
rotated credentials or macOS Keychain contents. Pattern detection complements
that check. Repeat scanning after adding new files or credentials.

Only sanitized review metadata is committed. Credential values, scanner binaries,
exported trees and raw scanner reports stay outside the repository in temporary
storage. No push or remote configuration was performed during the review.
