#!/bin/bash

# ==============================================================================
#  Super-Resolution Training Sweep Launcher (Auto-Adaptive Version)
#  功能：支持 1-3 个维度的参数调节。如果某维度留空，自动忽略。
# ==============================================================================

# 1. 基础设置
PYTHON="python"
TRAIN_SCRIPT="train.py"
BASE_CONFIG="config.yaml"
SWEEP_ROOT="exp/sweep_loss"

# ------------------------------------------------------------------------------
# 2. 参数扫描网格 (不想调节的参数，请将 VALS 设为空数组 ())
# ------------------------------------------------------------------------------

# --- 参数 1 ---
KEY1="loss.multi_res_weight.weight"
NAME1="mr"
VALS1=(180.0 250.0 300.0)  # 正常调节

# --- 参数 2 (示例：不想调节这个，留空) ---
KEY2="loss.gompsnr_weight.weight"
NAME2="gom"
VALS2=(100.0 200.0)             # <--- 留空，脚本会自动跳过此维度

# --- 参数 3 (示例：不想调节这个，留空) ---
KEY3="loss.recon_weight"
NAME3="rec"
VALS3=()             # <--- 留空，脚本会自动跳过此维度

# ==============================================================================
#  内部工具：Python YAML 修改器 (保持不变)
# ==============================================================================
update_yaml_py=$(cat <<EOF
import sys
import yaml
import os

def update_nested_dict(d, key_path, value):
    keys = key_path.split('.')
    current = d
    for k in keys[:-1]:
        current = current.setdefault(k, {})
    try:
        if value.lower() == 'true': val = True
        elif value.lower() == 'false': val = False
        elif '.' in value: val = float(value)
        else: val = int(value)
    except:
        val = value
    current[keys[-1]] = val

if __name__ == "__main__":
    src_path = sys.argv[1]
    dst_path = sys.argv[2]
    updates = sys.argv[3:]
    with open(src_path, 'r') as f:
        config = yaml.safe_load(f)
    for item in updates:
        k, v = item.split('=', 1)
        update_nested_dict(config, k, v)
    with open(dst_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
EOF
)

# ==============================================================================
#  逻辑预处理：处理空数组，防止循环不执行
# ==============================================================================

# 如果数组为空，赋予一个特殊标记 "__IGNORE__"，保证循环至少执行一次
[[ ${#VALS1[@]} -eq 0 ]] && VALS1=("__IGNORE__")
[[ ${#VALS2[@]} -eq 0 ]] && VALS2=("__IGNORE__")
[[ ${#VALS3[@]} -eq 0 ]] && VALS3=("__IGNORE__")

echo ">>> Starting Sweep..."
echo ">>> Base Config: $BASE_CONFIG"

# ==============================================================================
#  主循环
# ==============================================================================

for v1 in "${VALS1[@]}"; do
    for v2 in "${VALS2[@]}"; do
        for v3 in "${VALS3[@]}"; do
            
            # --- 动态构建参数列表和目录名 ---
            PY_ARGS=""
            EXP_ID_SUFFIX=""

            # 处理参数 1
            if [ "$v1" != "__IGNORE__" ]; then
                PY_ARGS="$PY_ARGS $KEY1=$v1"
                EXP_ID_SUFFIX="${EXP_ID_SUFFIX}_${NAME1}${v1}"
            fi

            # 处理参数 2
            if [ "$v2" != "__IGNORE__" ]; then
                PY_ARGS="$PY_ARGS $KEY2=$v2"
                EXP_ID_SUFFIX="${EXP_ID_SUFFIX}_${NAME2}${v2}"
            fi

            # 处理参数 3
            if [ "$v3" != "__IGNORE__" ]; then
                PY_ARGS="$PY_ARGS $KEY3=$v3"
                EXP_ID_SUFFIX="${EXP_ID_SUFFIX}_${NAME3}${v3}"
            fi

            # 如果没有定义任何参数，给一个默认名字
            if [ -z "$EXP_ID_SUFFIX" ]; then
                EXP_ID_SUFFIX="_default_run"
            fi
            
            # 去掉开头的下划线
            EXP_ID="${EXP_ID_SUFFIX:1}"
            EXP_DIR="${SWEEP_ROOT}/${EXP_ID}"

            # --- 以下是标准的执行逻辑 ---
            
            # 1. 准备目录
            if [ -d "$EXP_DIR" ]; then
                # 如果目录存在，跳过（或者你可以改为继续运行以触发断点续传）
                # 这里我们假设利用 train.py 的断点续传，所以不跳过，而是继续
                echo "[Check] Directory exists, attempting resume: $EXP_DIR"
            else
                mkdir -p "$EXP_DIR"
            fi

            echo "----------------------------------------------------------------"
            echo ">>> Running: $EXP_ID"
            
            # 2. 生成 Config
            TEMP_CONFIG="${EXP_DIR}/config.yaml"
            
            # 这里需要注意：Bash 变量扩展时如果不加引号，$PY_ARGS 会被拆分成多个参数传递，这正是我们想要的
            $PYTHON -c "$update_yaml_py" "$BASE_CONFIG" "$TEMP_CONFIG" \
                $PY_ARGS \
                "train.exp_dir=$EXP_DIR"

            # 3. 运行训练
            $PYTHON -u $TRAIN_SCRIPT "$TEMP_CONFIG"
            
            if [ $? -ne 0 ]; then
                echo ">>> FAILED: $EXP_ID"
            fi
            
        done
    done
done

echo "----------------------------------------------------------------"
echo ">>> All experiments finished."