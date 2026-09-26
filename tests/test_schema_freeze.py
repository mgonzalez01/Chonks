"""Pins the schema DDL. A DDL change needs a new hash here and a decision on SCHEMA_VERSION."""

import hashlib

from chonks.storage.schema import SCHEMA_DDL


def test_schema_ddl_sha256():
    assert hashlib.sha256(SCHEMA_DDL.encode()).hexdigest() == (
        "e7ad5fa57778901ff441eeaa18cecfba53a5243fd9673964e3c7e712277ba000"
    )
