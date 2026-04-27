# Notes

The ledger caught a real discovery: after adding category buckets, live rescoring exposed that explicit LedgerSession categories were being overwritten by description inference. That made the audit less honest, so the run grew to include category preservation and a regression test.
