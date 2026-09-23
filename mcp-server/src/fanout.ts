export interface BranchFailure {
  path: string;
  message: string;
}

// First line of a multi-path @subsystem search whose branches did not all answer.
export function partialSearchNote(name: string | null, total: number, failures: BranchFailure[]): string {
  if (failures.length === 0) return "";
  const detail = failures.map((f) => `${f.path}: ${f.message}`).join("; ");
  return `PARTIAL: search failed for ${failures.length} of ${total} paths of @${name} (${detail}); ` +
    "results cover the other paths only.\n\n";
}
