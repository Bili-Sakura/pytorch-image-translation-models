# Copyright (c) 2026 EarthBridge Team.
# Credits: Bosch Research LDDBM (Multimodal-Distribution-Translation-MDT).

"""MDT / LDDBM – training is done in the upstream repository.

For training, use the original MDT repository:

    git clone https://github.com/boschresearch/Multimodal-Distribution-Translation-MDT
    cd Multimodal-Distribution-Translation-MDT
    pip install -e .
    pip install -r requirements.txt

    # Super-resolution (16→128)
    python scripts/main.py --config_name sr --data_path lddbm/datasets/sr

    # Multi-view to 3D (ShapeNet)
    python scripts/main.py --config_name multi2shape --data_path lddbm/datasets/shapenet

See the upstream README for dataset preparation and full configuration.
"""
