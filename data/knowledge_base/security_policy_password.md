# Password and Account Security Policy

This document defines password and authentication requirements for all Bearn Analytics systems.

## Password requirements

All company account passwords must meet the following minimum requirements:

- At least **14 characters** long.
- A mix of uppercase, lowercase, numbers, and symbols.
- Must not match any of the employee's **last 10 passwords** (enforced automatically by Okta).
- Must be changed at least every **180 days**.

Passwords must never be shared, written down in plaintext, or reused across personal and work accounts.

## Multi-factor authentication

**Okta Verify MFA is mandatory** for all employees on all company systems, with no exceptions. Push notification is the default MFA method; SMS-based MFA is disabled company-wide due to SIM-swapping risk. Hardware security keys (YubiKey) are available on request for employees handling sensitive customer data.

## Password manager

All employees are provisioned a **1Password** account during onboarding and are required to use it for storing credentials to any work-related system. Password manager vaults are subject to the same 14-character minimum for the master password.

## Account lockout

After **5 consecutive failed login attempts**, an account is locked for 15 minutes. Repeated lockouts trigger an automatic security review and a notification to the IT security team.

## Reporting a compromised account

Suspected account compromise must be reported immediately to `security@bearnanalytics.example` and the `#security-incidents` Slack channel. Do not wait for confirmation before reporting.
