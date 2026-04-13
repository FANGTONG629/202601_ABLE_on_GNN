@echo off
setlocal enabledelayedexpansion

:: --- 配置列表 ---
set DATASET_LIST=ACM lastfm aug_citation
set RADIUS_LIST=0.1 0.3 0.5 0.7 0.9

:: --- 固定参数 ---
set DEVICE=0
set EPOCHS=25
set EXPLAIN=100
set NEIGHBOR=10
set RUNS=4
set HOPS=2
set L1=0.1
set L2=0.01

:: 初始化日志
echo Experiment Start Time: %date% %time% > experiment.log

:: 外层循环：遍历数据集
for %%d in (%DATASET_LIST%) do (

    :: --- 关键修改：根据数据集动态设置维度 ---
    if "%%d"=="ACM" (
        set DIM=64
    ) else (
        set DIM=128
    )

    :: 内层循环：遍历半径
    for %%r in (%RADIUS_LIST%) do (
        echo ---------------------------------------------------------- >> experiment.log 2>&1
        echo [%time%] Data: %%d ^| Radius: %%r ^| Dim: !DIM! >> experiment.log 2>&1
        echo ---------------------------------------------------------- >> experiment.log 2>&1

        :: 注意：在循环内部使用变量 DIM 时要用 !DIM!
        python ABLE_g_run.py ^
            --device_id %DEVICE% ^
            --dataset_name %%d ^
            --radius %%r ^
            --num_epochs %EPOCHS% ^
            --num_explain %EXPLAIN% ^
            --num_neighbor %NEIGHBOR% ^
            --num_runs %RUNS% ^
            --num_hops %HOPS% ^
            --emb_dim !DIM! ^
            --hidden_dim !DIM! ^
            --out_dim !DIM! ^
            --lambda_1 %L1% ^
            --lambda_2 %L2% >> experiment.log 2>&1
    )
)

echo.
echo ==========================================================
echo All experiments have finished. Check experiment.log for details.
echo ==========================================================
pause