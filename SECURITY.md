# Security Policy

## Reporting a Vulnerability

DO NOT OPEN A PUBLIC ISSUE TO REPORT A SECURITY VULNERABILITY

1.  **Primary Method:** Use the [built-in GitHub private vulnerability report](https://github.com/OWNER/REPO/security/advisories/new) for this repository.
2.  **Fallback Method:** Email reports to: g73447476@gmail.com
3.  **Encryption:** You may encrypt your report using our PGP key: [keys.openpgp.org/yourkey](https://keys.openpgp.org).

### Submission Requirements
To ensure a rapid response, please include the following in your report:
* A descriptive title and the type of vulnerability.
* The specific commit hash or branch affected.
* A detailed proof-of-concept (PoC) or clear steps to reproduce the issue.
* The potential impact of the vulnerability.

### Our Disclosure Commitment
1.  **Acknowledgement:** We will acknowledge receipt of your report within 3 business days.
2.  **Remediation:** We aim to patch critical severity issues as a priority in the next push to the main branch.
3.  **Credit:** We will publicly credit you in the release notes once the project moves to a stable version, unless you request anonymity.

## Scope

### Eligible Vulnerabilities
* Remote code execution (RCE)
* Authentication or authorisation bypass
* Privilege escalation
* Data leakage of sensitive information
* Injection flaws

### Out of Scope
* Theoretical attacks without a working proof-of-concept.
* Social engineering of maintainers or users.
* Vulnerabilities in third-party dependencies already publicly disclosed.
* Issues requiring the attacker to already possess root-level access to the host.

## Safe Harbour
+We support security research conducted on this project in good faith. We will not take legal action or involve law enforcement for research that complies with this policy.

## Bug Bounty
This project does not operate a paid bug bounty program.
