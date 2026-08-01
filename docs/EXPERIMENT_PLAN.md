# Historical ISRA experiment plan

This plan is retained only to explain what was tested and why the legacy
iterative runtime was removed. It is superseded by
[REPLACEMENT_DESIGN.md](REPLACEMENT_DESIGN.md).

The controlled Llama 3.1 8B mechanism smoke answered its immediate question:
the grounded one-repair and equal-control retry variants produced zero paired
fixes at positive latency. The remaining implementation cannot be presented as
an improvement layer.

The active plan is now:

1. Keep the legacy negative result and evaluator integrity work in the record.
2. Test one frozen one-pass mechanism at a time.
3. Use a small development screen only as a kill/proceed gate.
4. After one positive screen, freeze parameters and evaluate a disjoint,
   date-filtered primary benchmark.
5. Prefer a stronger clean backbone or execution-verified correction
   distillation over new same-model review prompts if SPA fails confirmation.
