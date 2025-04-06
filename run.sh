#!/bin/bash

# 设置默认值
ENV_PATH="/home/kosmos/calibre-web-env"
DATA_PATH="/home/kosmos/.calibre-web"
MODE="production"  # 默认为生产环境

# 显示使用方法
usage() {
    echo "使用方法: $0 [-e env_path] [-d data_path] [-m mode]"
    echo "选项:"
    echo "  -e: 虚拟环境路径 (默认: ${ENV_PATH})"
    echo "  -d: 数据目录路径 (默认: ${DATA_PATH})"
    echo "  -m: 运行模式 (development/production, 默认: ${MODE})"
    exit 1
}

# 处理命令行参数
while getopts "e:d:m:h" opt; do
    case $opt in
        e) ENV_PATH="$OPTARG";;
        d) DATA_PATH="$OPTARG";;
        m) MODE="$OPTARG";;
        h) usage;;
        ?) usage;;
    esac
done

# 验证运行模式
if [[ "$MODE" != "development" && "$MODE" != "production" ]]; then
    echo "错误: 运行模式必须是 'development' 或 'production'"
    exit 1
fi

# 检查虚拟环境目录是否存在
if [ ! -d "$ENV_PATH" ]; then
    echo "错误: 虚拟环境目录不存在: $ENV_PATH"
    exit 1
fi

# 检查数据目录是否存在
if [ ! -d "$DATA_PATH" ]; then
    echo "错误: 数据目录不存在: $DATA_PATH"
    exit 1
fi

# 激活虚拟环境
source "$ENV_PATH/bin/activate"

# 设置环境变量
export CALIBRE_DBPATH="$DATA_PATH"
export APP_MODE="$MODE"

# 输出当前配置
echo "----------------------------------------"
echo "Calibre-Web 启动配置:"
echo "虚拟环境: $ENV_PATH"
echo "数据目录: $DATA_PATH"
echo "运行模式: $MODE"
echo "----------------------------------------"

# 启动应用
python3 cps.py
