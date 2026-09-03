# Secrets

Mounted read-only into the container for automatic Slurm token renewal
(`SLURM_TOKEN_MODE=command`). Everything here is gitignored.

## `slurm_token_key`

An SSH private key that can run `scontrol token` on the token host.

**This key can mint Slurm credentials.** Constrain it on the remote side so a
stolen copy cannot open a shell — add a forced command to that account's
`~/.ssh/authorized_keys`:

```
command="scontrol token lifespan=3600",no-port-forwarding,no-agent-forwarding,no-pty,no-X11-forwarding ssh-ed25519 AAAAC3Nza... scheduler-token-renewal
```

Generate a dedicated key — do not reuse a personal one:

```bash
ssh-keygen -t ed25519 -N '' -f secrets/slurm_token_key -C scheduler-token-renewal
chmod 600 secrets/slurm_token_key
sudo chown 1000:1000 secrets/slurm_token_key   # the container's `scheduler` user
```

## `known_hosts`

Required. Without it, host key checking cannot verify we are talking to the
real token host, and an impersonator could return a token pointing at their own
slurmrestd.

```bash
ssh-keyscan titan > secrets/known_hosts
```

## Prefer `SLURM_TOKEN_MODE=file` where you can

If something outside the app can refresh a token file (cron, systemd timer, a
Kubernetes secret), use that instead. The scheduler then holds only a token —
never a credential capable of minting more.
