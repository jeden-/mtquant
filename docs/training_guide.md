# MTQuant Training Guide

## End-to-End 3-Phase Training

### Overview

MTQuant uses a hierarchical 3-phase training approach:

1. **Phase 1**: Individual Specialist Training
   - Train Forex Specialist (EURUSD, GBPUSD, USDJPY)
   - Train Commodities Specialist (XAUUSD, WTIUSD)
   - Train Equity Specialist (SPX500, NAS100, US30)

2. **Phase 2**: Meta-Controller Training
   - Train portfolio-level decision maker
   - Learn capital allocation strategies
   - Learn risk appetite management

3. **Phase 3**: Joint Training
   - Fine-tune all agents together
   - Gradient coordination
   - Portfolio-level optimization

---

## Quick Start

### Run Complete Training

```bash
# Full training (all 3 phases)
python scripts/run_end_to_end_training.py

# Quick test (reduced timesteps for testing)
python scripts/run_end_to_end_training.py --quick-test

# Custom timesteps
python scripts/run_end_to_end_training.py \
  --phase1-timesteps 100000 \
  --phase2-timesteps 50000 \
  --phase3-timesteps 50000

# Parallel training (Phase 1)
python scripts/run_end_to_end_training.py --parallel
```

### Run Individual Phases

```bash
# Only Phase 1
python scripts/run_end_to_end_training.py --phases 1

# Phase 1 and 2
python scripts/run_end_to_end_training.py --phases 1,2

# Only Phase 2 (requires Phase 1 results)
python scripts/run_end_to_end_training.py --phases 2
```

---

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--phase1-timesteps` | 100000 | Timesteps per specialist (Phase 1) |
| `--phase2-timesteps` | 50000 | Timesteps for meta-controller (Phase 2) |
| `--phase3-timesteps` | 50000 | Timesteps for joint training (Phase 3) |
| `--parallel` | False | Train specialists in parallel |
| `--save-freq` | 10000 | Model save frequency (timesteps) |
| `--phases` | "1,2,3" | Phases to run (comma-separated) |
| `--quick-test` | False | Quick test mode (1000/500/500 timesteps) |

---

## Training Time Estimates

### Full Training (Production)
- **Phase 1**: ~8-12 hours (100K timesteps per specialist)
- **Phase 2**: ~4-6 hours (50K timesteps)
- **Phase 3**: ~4-6 hours (50K timesteps)
- **Total**: ~16-24 hours

### Quick Test Mode
- **Phase 1**: ~5-10 minutes (1K timesteps per specialist)
- **Phase 2**: ~2-3 minutes (500 timesteps)
- **Phase 3**: ~2-3 minutes (500 timesteps)
- **Total**: ~10-15 minutes

---

## Output Files

### Logs
```
logs/e2e_training/
├── e2e_training_YYYYMMDD_HHMMSS.log
├── phase1_results.json
├── phase2_results.json
├── phase3_results.json
└── training_summary.json
```

### Models
```
models/
├── specialists/
│   ├── forex_specialist.zip
│   ├── commodities_specialist.zip
│   └── equity_specialist.zip
├── meta_controller/
│   └── meta_controller.zip
└── checkpoints/
    ├── checkpoint_10000/
    ├── checkpoint_20000/
    └── ...
```

### Metrics
```
metrics/training/
├── phase1_metrics.csv
├── phase2_metrics.csv
└── phase3_metrics.csv
```

---

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/tensorboard

# Open browser
http://localhost:6006
```

### Real-time Logs

```bash
# Follow training logs
tail -f logs/e2e_training/e2e_training_*.log

# Watch Phase 1 progress
watch -n 5 'tail -n 20 logs/e2e_training/e2e_training_*.log'
```

---

## Configuration

### Agent Configuration

Edit `config/agents.yaml`:

```yaml
specialists:
  forex:
    symbols: [EURUSD, GBPUSD, USDJPY]
    learning_rate: 0.0003
    batch_size: 64
    
  commodities:
    symbols: [XAUUSD, WTIUSD]
    learning_rate: 0.0003
    batch_size: 64
    
  equity:
    symbols: [SPX500, NAS100, US30]
    learning_rate: 0.0003
    batch_size: 64

meta_controller:
  learning_rate: 0.0001
  batch_size: 32
```

---

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size in config/agents.yaml
batch_size: 32  # instead of 64

# Train specialists sequentially (not parallel)
python scripts/run_end_to_end_training.py  # without --parallel
```

### Training Too Slow

```bash
# Use GPU if available
export CUDA_VISIBLE_DEVICES=0

# Reduce timesteps for testing
python scripts/run_end_to_end_training.py --quick-test
```

### Resume Training

```bash
# Phase 1 results are saved, skip to Phase 2
python scripts/run_end_to_end_training.py --phases 2,3
```

---

## Validation

### After Training

1. **Check Model Files**
   ```bash
   ls -lh models/specialists/
   ls -lh models/meta_controller/
   ```

2. **Review Training Summary**
   ```bash
   cat logs/e2e_training/training_summary.json
   ```

3. **Analyze Metrics**
   ```python
   import json
   
   with open('logs/e2e_training/phase1_results.json') as f:
       results = json.load(f)
   
   print(f"Specialists trained: {len(results['specialists'])}")
   ```

---

## Next Steps

After training:

1. **Paper Trading** - Test models on demo account (30 days)
2. **Performance Analysis** - Analyze Sharpe ratio, drawdown, win rate
3. **Model Tuning** - Adjust hyperparameters based on results
4. **Live Deployment** - Deploy to production (start with 10% capital)

---

## Support

For issues or questions:
- Check logs: `logs/e2e_training/`
- Review documentation: `docs/`
- GitHub Issues: https://github.com/jeden-/mtquant/issues

---

**Last Updated:** October 15, 2025  
**Version:** 1.0.0



