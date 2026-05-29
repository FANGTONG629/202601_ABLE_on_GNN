@echo off
setlocal enabledelayedexpansion

:: --- 这是用来测稀疏度的脚本 ---
:: --- 配置列表 ---
set DATASET_LIST=lastfm aug_citation ACM
set Lambda1_LIST= 0.001 0.01 0.1 0.3
set Lambda2_LIST= 0.001 0.01 0.1 0.3

:: --- 固定参数 ---
set DEVICE=0
set EPOCHS=25
set EXPLAIN=100
set NEIGHBOR=10
set RUNS=1
set HOPS=2
set RADIUS=0.3


:: 初始化日志
echo Experiment Start Time: %date% %time% > experiment3.log

:: 外层循环：遍历数据集
for %%d in (%DATASET_LIST%) do (

    :: --- 关键修改：根据数据集动态设置维度 ---
    if "%%d"=="ACM" (
        set DIM=64
    ) else (
        set DIM=128
    )

    :: 内层循环：遍历半径
    for %%i in (%Lambda1_LIST%) do (
        for %%j in (%Lambda2_LIST%) do (
            echo ---------------------------------------------------------- >> experiment3.log 2>&1
            echo [%time%] Data: %%d ^| Lambda1: %%i ^|  Lambda2: %%j ^| Dim: !DIM! >> experiment3.log 2>&1
            echo ---------------------------------------------------------- >> experiment3.log 2>&1

            :: 注意：在循环内部使用变量 DIM 时要用 !DIM!
            python ABLE_g_run.py ^
                --device_id %DEVICE% ^
                --dataset_name %%d ^
                --radius %RADIUS% ^
                --num_epochs %EPOCHS% ^
                --num_explain %EXPLAIN% ^
                --num_neighbor %NEIGHBOR% ^
                --num_runs %RUNS% ^
                --num_hops %HOPS% ^
                --emb_dim !DIM! ^
                --hidden_dim !DIM! ^
                --out_dim !DIM! ^
                --save_explanation ^
                --save_excel ^
                --lambda_1 %%i ^
                --lambda_2 %%j >> experiment3.log 2>&1
        )
    )
)

echo.
echo ==========================================================
echo All experiments have finished. Check experiment3.log for details.
echo ==========================================================
pause


