# Process data

```bash
python colbert/1.1_data.py -i 0
```

# Train

```bash
python run.py
```

# Evaluate

```bash
python run.py --mode val --ckpt <path_to_checkpoint>
```