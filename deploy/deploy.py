"""Deploy runtime sources using environment settings supplied by GitHub secrets."""

import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("VDS_HOST", "VDS_USER", "VDS_DEPLOY_DIR", "VDS_DEPLOY_KEY", "VDS_KNOWN_HOSTS",
            "SERVICE_HOST", "BASIC_AUTH_USER", "BASIC_AUTH_HASH")


def load_settings(environment):
    settings = {name: environment.get(name, "") for name in REQUIRED}
    for name, value in settings.items():
        if not value.strip():
            raise ValueError(f"Missing {name}")
    patterns = {"VDS_HOST": r"[A-Za-z0-9][A-Za-z0-9.-]*",
                "VDS_USER": r"[A-Za-z0-9_][A-Za-z0-9_-]*",
                "VDS_DEPLOY_DIR": r"/[A-Za-z0-9._/-]+",
                "SERVICE_HOST": r"[A-Za-z0-9][A-Za-z0-9.-]*",
                "BASIC_AUTH_USER": r"[A-Za-z0-9_-]+",
                "BASIC_AUTH_HASH": r"\$2[aby]\$\d{2}\$[A-Za-z0-9./]{53}"}
    for name, pattern in patterns.items():
        if not re.fullmatch(pattern, settings[name]):
            raise ValueError(f"Invalid {name}")
    directory = PurePosixPath(settings["VDS_DEPLOY_DIR"])
    if ".." in directory.parts or len(directory.parts) < 3:
        raise ValueError("Invalid VDS_DEPLOY_DIR")
    port = environment.get("VDS_PORT") or "22"
    if not port.isascii() or not port.isdecimal() or not 1 <= int(port) <= 65535:
        raise ValueError("Invalid VDS_PORT")
    settings["VDS_PORT"] = port
    return settings


def private_environment(settings):
    return "".join(f"{name}='{settings[name]}'\n" for name in
                   ("SERVICE_HOST", "BASIC_AUTH_USER", "BASIC_AUTH_HASH"))


def main():
    settings = load_settings(os.environ)
    with TemporaryDirectory(prefix="private-llm-deploy-") as temporary:
        work = Path(temporary)
        for name, value in (("key", settings["VDS_DEPLOY_KEY"]),
                            ("known_hosts", settings["VDS_KNOWN_HOSTS"]),
                            ("private.env", private_environment(settings))):
            path = work / name
            path.write_text(value.rstrip("\n") + "\n")
            path.chmod(0o600)
        ssh = ["ssh", "-i", str(work / "key"), "-p", settings["VDS_PORT"],
               "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes",
               "-o", f"UserKnownHostsFile={work / 'known_hosts'}", "-o", "ConnectTimeout=15",
               "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3"]
        target = settings["VDS_USER"] + "@" + settings["VDS_HOST"]
        directory = settings["VDS_DEPLOY_DIR"].rstrip("/")
        def remote(command):
            subprocess.run(ssh + [target, command], check=True)
        remote("mkdir -p " + " ".join(shlex.quote(directory + suffix) for suffix in
               ("/app", "/static", "/deploy", "/data/models", "/data/document_index", "/data/documents")))
        rsync = ["rsync", "-az", "-e", shlex.join(ssh)]
        # Delete stale runtime files only inside these two source directories.
        # SQLite volumes, model weights and deployment credentials are preserved.
        for name in ("app", "static"):
            subprocess.run(rsync + ["--delete", "--exclude=__pycache__/", str(ROOT / name) + "/",
                                    f"{target}:{directory}/{name}/"], check=True)
        subprocess.run(rsync + [str(ROOT / name) for name in
                       ("Dockerfile", ".dockerignore", "compose.yaml", "requirements.txt", "requirements-indexing.txt")]
                       + [f"{target}:{directory}/"], check=True)
        subprocess.run(rsync + [str(ROOT / "deploy/Caddyfile"), str(work / "private.env"),
                                f"{target}:{directory}/deploy/"], check=True)
        remote(f"cd {shlex.quote(directory)} && "
               "sudo -n docker compose --env-file deploy/private.env config --quiet && "
               "sudo -n docker compose --env-file deploy/private.env up -d --build --wait --wait-timeout 600 && "
               # A file bind mount can retain the old inode after rsync; Caddy
               # also needs an explicit reload to apply changed configuration.
               "sudo -n docker compose --env-file deploy/private.env up -d --no-deps "
               "--force-recreate --wait --wait-timeout 60 gateway")


if __name__ == "__main__":
    main()
