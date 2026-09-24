"""Pins the schema DDL. A DDL change needs a new hash here and a decision on SCHEMA_VERSION."""

import hashlib

from chonks.storage.schema import SCHEMA_DDL


def test_schema_ddl_sha256():
    assert hashlib.sha256(SCHEMA_DDL.encode()).hexdigest() == (
        "a655cafe7b9587d649f4c697d8c2b8c037ae95364ce9d82a9f3afcd2e3e48ce5"
    )
