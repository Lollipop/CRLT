Here we provide some usage examples, you can easily modify and combine according to their needs:

### SimCSE
```bash
CUDA_VISIBLE_DEVICES=0 python main.py examples/simcse.json
```

### SimCSE-NLI
```bash
CUDA_VISIBLE_DEVICES=0 python main.py examples/simcse_nli.json
```
### Memory Bank
```bash
CUDA_VISIBLE_DEVICES=0 python main.py examples/memory_bank.json
```
### MoCo
```bash
CUDA_VISIBLE_DEVICES=0 python main.py examples/moco.json
```

### Data Augmentation Methods
Refer to the `augmentation_method.json` file to modify the corresponding configuration items, query and key samples can be created by different data enhancement methods:
| Method | Configuration Parameter |
|:--------|:-----------:|
| Delete | `granularity`(word \| span), `probability`(float) |
| Insert | `granularity`(word), `number`(int) |
| Replace | `granularity`(word), `number`(int) |
| BackTranslation | `granularity`(semantic), `device`(int) |
| Paraphrase | `granularity`(semantic), `device`(int) |
| Mask | `granularity`(word \| span \| feature), `probability`(float) |
| Shuffle | `granularity`(word \| span \| feature), `number`(int) |
