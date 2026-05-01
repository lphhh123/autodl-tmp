# CH3 HALP + No-Probe Fix Commands

## One-shot parallel run on 6 GPUs
```bash
cd proj_ast2_ucf101_full
mkdir -p $HOME/runlogs_A
setsid bash scripts/launch_ch3_halpnoprobe_fix_6gpu.sh > $HOME/runlogs_A/ch3_halpnoprobe_fix.log 2>&1 < /dev/null &
tail -f $HOME/runlogs_A/ch3_halpnoprobe_fix.log
```

## Verify strict protocol
```bash
grep -E "start_outer=15|\[CH3 STRICT\]|\[ERROR\]\[CH3 STRICT\]" -n outputs/stdout_agg/EXP-A2p25-halp-k90-*_seed0.log | head
grep -E "start_outer=15|\[CH3 STRICT\]|\[ERROR\]\[CH3 STRICT\]" -n outputs/stdout_agg/EXP-A2p25-ab-nolookahead-k90-*_seed0.log | head
```
