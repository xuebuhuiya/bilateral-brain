@echo off
:: 使用 UTF-8 编码避免乱码
@chcp 65001 >nul

:: 启用延迟变量扩展
setlocal enabledelayedexpansion

:: 定义种子数组
set "seeds[0]=6"
set "seeds[1]=7"
set "seeds[2]=8"
set "seeds[3]=9"
set "seeds[4]=10"

:: 定义 concat_method 数组
set "concat_method[0]=cat"
set "concat_method[1]=add"
set "concat_method[2]=halfdata"
set "concat_method[3]=ave_pooling"

:: 定义 fine 模型路径数组
set "fine_model_path[0]=E:\Desktop\research\bilateral-brain\runs\B1_five_fine-unilateral-resnet9-fine\20240929200807-seed6\checkpoints\epoch=50-val_acc=0.668.ckpt"
set "fine_model_path[1]=E:\Desktop\research\bilateral-brain\runs\B1_five_fine-unilateral-resnet9-fine\20240929202824-seed7\checkpoints\epoch=50-val_acc=0.665.ckpt"
set "fine_model_path[2]=E:\Desktop\research\bilateral-brain\runs\B1_five_fine-unilateral-resnet9-fine\20240929204723-seed8\checkpoints\epoch=55-val_acc=0.670.ckpt"
set "fine_model_path[3]=E:\Desktop\research\bilateral-brain\runs\B1_five_fine-unilateral-resnet9-fine\20240929210623-seed9\checkpoints\epoch=56-val_acc=0.674.ckpt"
set "fine_model_path[4]=E:\Desktop\research\bilateral-brain\runs\B1_five_fine-unilateral-resnet9-fine\20240929212532-seed10\checkpoints\epoch=53-val_acc=0.667.ckpt"

:: 定义 coarse 模型路径数组
set "coarse_model_path[0]=E:\Desktop\research\bilateral-brain\runs\B1_five_coarse-unilateral-resnet9-coarse\20240929190218-seed6\checkpoints\epoch=35-val_acc=0.738.ckpt"
set "coarse_model_path[1]=E:\Desktop\research\bilateral-brain\runs\B1_five_coarse-unilateral-resnet9-coarse\20240929191534-seed7\checkpoints\epoch=34-val_acc=0.736.ckpt"
set "coarse_model_path[2]=E:\Desktop\research\bilateral-brain\runs\B1_five_coarse-unilateral-resnet9-coarse\20240929192810-seed8\checkpoints\epoch=32-val_acc=0.728.ckpt"
set "coarse_model_path[3]=E:\Desktop\research\bilateral-brain\runs\B1_five_coarse-unilateral-resnet9-coarse\20240929194028-seed9\checkpoints\epoch=35-val_acc=0.732.ckpt"
set "coarse_model_path[4]=E:\Desktop\research\bilateral-brain\runs\B1_five_coarse-unilateral-resnet9-coarse\20240929195245-seed10\checkpoints\epoch=34-val_acc=0.731.ckpt"

:: 循环遍历 concat_method
for /L %%j in (0,1,3) do (
    :: 循环执行五次实验
    for /L %%i in (0,1,4) do (
        echo 正在使用 concat_method=!concat_method[%%j]! 和种子 !seeds[%%i]! 进行实验

        :: 生成临时配置文件，替换种子、模型路径和 concat_method
        copy "configs\config.yaml" "configs\config_tmp.yaml"

        :: 使用 PowerShell 进行字符串替换
        powershell -Command "(Get-Content 'configs/config_tmp.yaml') -replace 'seeds: \[.*\]', 'seeds: [!seeds[%%i]!]' | Set-Content 'configs/config_tmp.yaml'"
        powershell -Command "(Get-Content 'configs/config_tmp.yaml') -replace 'model_path_fine:.*', 'model_path_fine: !fine_model_path[%%i]!' | Set-Content 'configs/config_tmp.yaml'"
        powershell -Command "(Get-Content 'configs/config_tmp.yaml') -replace 'model_path_coarse:.*', 'model_path_coarse: !coarse_model_path[%%i]!' | Set-Content 'configs/config_tmp.yaml'"
        powershell -Command "(Get-Content 'configs/config_tmp.yaml') -replace 'concat_method:.*', 'concat_method: !concat_method[%%j]!' | Set-Content 'configs/config_tmp.yaml'"

        :: 替换 exp_name 中的 concat_method
        powershell -Command "(Get-Content 'configs/config_tmp.yaml') -replace 'exp_name:.*', 'exp_name: B3_conv_!concat_method[%%j]!' | Set-Content 'configs/config_tmp.yaml'"

        :: 运行实验
        python trainer.py --config=configs/config_tmp.yaml
    )
)

pause
