# New-Hire Laptop Provisioning

This guide covers how new employees receive and set up their work laptop.

## Default hardware

All new hires are issued a **MacBook Pro 14" (M-series)** as their default machine. Employees on the Engineering team working on Windows-specific tooling may request a Windows laptop instead via a ticket to the IT Helpdesk; approval is not automatic and requires manager sign-off.

Laptops are refreshed on a **3-year cycle**, or sooner if hardware fails or no longer meets role requirements.

## First-day setup

1. Sign in to the laptop using your **Okta SSO** credentials, provided in your welcome email.
2. Open a terminal and run the onboarding script: `bearn-setup.sh`. This installs standard tooling (Git, the company VPN client, Slack, 1Password, and department-specific software based on your role).
3. Enroll the device in the company's Mobile Device Management (MDM) system when prompted — this is mandatory for all company-owned hardware.
4. Verify Okta Verify (the MFA app) is installed and paired before attempting to access internal systems.

## Support

For setup issues, contact the IT Helpdesk via the `#it-helpdesk` Slack channel or the internal ticketing portal. Standard response time is within 4 business hours.

## Loaner and travel equipment

Loaner laptops are available from the IT desk for employees whose primary device is being repaired. International travel adapters and a second monitor for home offices can be requested through the same ticketing portal.
