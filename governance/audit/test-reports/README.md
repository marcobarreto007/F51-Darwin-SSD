# Source audit reports

Run the complete CPU-safe source audit from a clean commit with:

```powershell
& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoProfile -ExecutionPolicy Bypass `
  -File tools\run_gold_source_audit.ps1
```

The detailed report is written atomically below the ignored
`workspace/runtime/evaluations/` tree. Only a redacted, deterministic summary
bound to a frozen source commit may be copied into this tracked directory.

A passing report requires every recorded command to exit zero, a clean tracked
worktree, and identical `source_commit` and `report_source_commit`. Hardware,
checkpoint and corpus verification are separate later gates; this report alone
does not establish GOLD status.
