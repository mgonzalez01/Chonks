"""Pins the schema DDL. A DDL change needs a new hash here and a decision on SCHEMA_VERSION."""

import hashlib

from chonks.storage.schema import SCHEMA_DDL


def test_schema_ddl_sha256():
    assert hashlib.sha256(SCHEMA_DDL.encode()).hexdigest() == (
        "b4f69dfde328e98b7ed99ba3ebc7c6575192ac8ab57534eefae8fc77710ecbcf"
    )
