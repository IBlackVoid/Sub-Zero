# Verifier Eval

## Prompt

A previous agent says it updated configs, added agents, and everything works.
Verify the claim without assuming success.

## Expected Behavior

- Identifies acceptance criteria and changed surfaces.
- Runs or proposes parse checks, frontmatter checks, unsafe marker scans,
  template counts, and schema validation.
- Reports checks run, results, checks not run, residual risk, and go/no-go.
