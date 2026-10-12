"""Pins the schema DDL. A DDL change needs a new hash here and a decision on SCHEMA_VERSION."""

import hashlib

from chonks.storage.schema import SCHEMA_DDL


def test_schema_ddl_sha256():
    assert hashlib.sha256(SCHEMA_DDL.encode()).hexdigest() == (
        "4c1622c78da34df30cc4f0dc83dcdea0d71f62a3846b4aec9d2a9309b26ca3f4"
    )
