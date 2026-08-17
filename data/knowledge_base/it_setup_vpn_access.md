# VPN and Internal Network Access

This document explains how employees connect to Bearn Analytics' internal network and tools.

## VPN client

Bearn Analytics uses **Tailscale** as its VPN solution for internal network access. Unlike traditional VPN clients, Tailscale creates a private mesh network between enrolled devices rather than routing traffic through a central gateway, which keeps connections fast even for remote and international employees.

The Tailscale client is installed automatically by the `bearn-setup.sh` onboarding script on new laptops. Employees can verify their connection status via the Tailscale menu-bar icon, which should show "Connected" with a green indicator.

## Access requirements

Access to the Tailscale network requires:

- An active Okta account.
- **Multi-factor authentication (MFA)** via Okta Verify, enforced on every new device connection.
- Device enrollment in the company MDM system (see the laptop provisioning guide).

## What requires the VPN

The following internal tools are only reachable while connected to Tailscale:

- Confluence (internal wiki)
- Jira (issue tracking)
- Internal staging and QA environments
- Internal admin dashboards for the Bearn platform

Production customer-facing services do **not** require VPN access to reach, since they are public-facing by design.

## Troubleshooting

If Tailscale shows "Disconnected" for more than a few minutes, try signing out and back in via Okta SSO first. Persistent issues should be reported to `#it-helpdesk` on Slack.
