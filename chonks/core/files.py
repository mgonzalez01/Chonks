"""File-table vocabulary shared by the indexer and the query layer."""

# Stored as the content hash of a file that lost a chunk to a server error. No
# file hashes to it, so the next run indexes the file again.
INCOMPLETE_FILE_HASH = "incomplete"
