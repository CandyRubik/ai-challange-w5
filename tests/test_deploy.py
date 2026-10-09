from io import StringIO

from dotenv import dotenv_values
import pytest

from deploy.deploy import load_settings, private_environment


def environment():
    return dict(VDS_HOST="vds.example.com", VDS_USER="deploy", VDS_PORT="2222",
                VDS_DEPLOY_DIR="/srv/llm", VDS_DEPLOY_KEY="private-key\n",
                VDS_KNOWN_HOSTS="[vds.example.com]:2222 ssh-ed25519 public-key\n",
                SERVICE_HOST="llm.example.com", BASIC_AUTH_USER="test",
                BASIC_AUTH_HASH="$2a$14$" + "a" * 53)


def test_deployment_environment_preserves_auth_hash_and_excludes_ssh_credentials():
    settings = load_settings(environment())
    configured = dotenv_values(stream=StringIO(private_environment(settings)))
    assert configured == {name: settings[name] for name in
                          ("SERVICE_HOST", "BASIC_AUTH_USER", "BASIC_AUTH_HASH")}


@pytest.mark.parametrize("key,value", [
    ("VDS_HOST", ""), ("VDS_HOST", "host; command"), ("VDS_USER", "user name"),
    ("VDS_PORT", "70000"), ("VDS_DEPLOY_DIR", "/"), ("VDS_DEPLOY_DIR", "/srv/../llm"),
    ("SERVICE_HOST", "host\nOTHER=value"), ("BASIC_AUTH_HASH", "not-a-bcrypt-hash"),
])
def test_bad_deployment_settings_are_rejected_before_connecting(key, value):
    with pytest.raises(ValueError, match=key):
        load_settings({**environment(), key: value})
