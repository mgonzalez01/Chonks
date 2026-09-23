"""Pins the schema DDL. A DDL change needs a new hash here and a decision on SCHEMA_VERSION."""

import hashlib

from chonks.storage.schema import SCHEMA_DDL


def test_schema_ddl_sha256():
    assert hashlib.sha256(SCHEMA_DDL.encode()).hexdigest() == (
        "7c12e00c93d63681f0b3e23f33ff3ee81a9e216aa614f3ffd80e04db9c49d7b9"
    )
