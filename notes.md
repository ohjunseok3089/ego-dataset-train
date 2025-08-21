== Train ==
python src/csts_integration/csts_baseline.py \
    --init_method tcp://localhost:9880 \
    --cfg configs/CSTS_Ego4D_Gaze_Forecast.yaml \
    TRAIN.BATCH_SIZE 8 \
    TEST.ENABLE False \
    NUM_GPUS 2 \
    DATA.PATH_PREFIX=/mas/robots/prg-ego4d/raw/v2/clips.gaze \
    TRAIN.CHECKPOINT_FILE_PATH ~/junseok/K400_MVIT_B_16x4_CONV.pyth \
    OUTPUT_DIR=/mas/robots/prg-ego4d/out/csts_ego4d \
    MODEL.LOSS_FUNC kldiv+egonce \
    MODEL.LOSS_ALPHA 0.05 \
    RNG_SEED 21
==============================================================
== Test ==
== head_orientation ==
python src/csts_integration/csts_baseline.py \
    --init_method tcp://localhost:9880 \
    --cfg configs/CSTS_Ego4D_Gaze_Head_Orientation.yaml \
    TRAIN.BATCH_SIZE 8 \
    TRAIN.ENABLE False \
    NUM_GPUS 4 \
    DATA.PATH_PREFIX /mas/robots/prg-ego4d/raw/v2/clips.gaze \
    TRAIN.CHECKPOINT_FILE_PATH ~/junseok/checkpoint/csts_ego4d_forecast.pyth \
    OUTPUT_DIR /mas/robots/prg-ego4d/out/csts_ego4d \
    MODEL.LOSS_FUNC head_orientation \
    MODEL.LOSS_ALPHA 0.05 \
    RNG_SEED 21

== orignial == 
python src/csts_integration/csts_baseline.py \
    --cfg configs/CSTS_Ego4D_Gaze_Forecast.yaml \
    TRAIN.ENABLE False \
    TEST.BATCH_SIZE 8 \
    NUM_GPUS 4 \
    DATA.PATH_PREFIX /mas/robots/prg-ego4d/raw/v2/clips.gaze \
    TEST.CHECKPOINT_FILE_PATH ~/junseok/checkpoint/csts_ego4d_forecast.pyth \
    OUTPUT_DIR /mas/robots/prg-ego4d/out/csts_ego4d_evaluation \
    RNG_SEED 21