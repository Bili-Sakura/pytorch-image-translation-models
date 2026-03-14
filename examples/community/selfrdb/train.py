# Copyright (c) 2026 EarthBridge Team.
# Credits: SelfRDB (Arslan et al., Medical Image Analysis 2024).

"""Training harness for SelfRDB.

SelfRDB uses Lightning and its own training script. Run training in the
original repository:

  cd /path/to/SelfRDB
  python main.py fit --config config.yaml \\
      --trainer.logger.name $EXP_NAME \\
      --data.dataset_dir $DATA_DIR \\
      --data.source_modality $SOURCE \\
      --data.target_modality $TARGET \\
      ...

See https://github.com/icon-lab/SelfRDB for full training instructions.
"""
