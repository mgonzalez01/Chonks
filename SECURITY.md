# Security

Chonks runs an HTTP API and an MCP adapter with no authentication. Both bind to loopback by default, and the documentation says where that is not the case. Exposing either beyond a trusted network is a deployment decision, not a bug.

To report a vulnerability in the code itself, use GitHub's private vulnerability reporting on this repository (the Security tab). Please do not open a public issue for it.
