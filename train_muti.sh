


train_model() {
    CUDA_VISIBLE_DEVICES=0 python /home/sonel/code/Sonel_code/OOD_SOTA/main.py --config_file $1 --phase train
}


configs=(




"/home/sonel/code/Sonel_code/OOD_SOTA/Configs/FPNBBALSK/DOTA_FPNLSKBBA_r152.yaml"
"/home/sonel/code/Sonel_code/OOD_SOTA/Configs/FPNBBALSK/DOTA_FPNLSKBBA_r152.yaml"
"/home/sonel/code/Sonel_code/OOD_SOTA/Configs/FPNBBALSK/DOTA_FPNLSKBBA_r152.yaml"
"/home/sonel/code/Sonel_code/OOD_SOTA/Configs/FPNBBALSK/DOTA_FPNLSKBBA_r152.yaml"
"/home/sonel/code/Sonel_code/OOD_SOTA/Configs/FPNBBALSK/DOTA_FPNLSKBBA_r152.yaml"
"/home/sonel/code/Sonel_code/OOD_SOTA/Configs/FPNBBALSK/DOTA_FPNLSKBBA_r152.yaml"
)


for config in "${configs[@]}"; do
    echo "Training model with config: $config"
    train_model "$config"
    if [ $? -ne 0 ]; then
        echo "█ █ █ █ Training failed for config: $config  █ █ █ █"

    fi

    echo "----------------------------------------------------------------------"
done

echo "All models trained successfully."
