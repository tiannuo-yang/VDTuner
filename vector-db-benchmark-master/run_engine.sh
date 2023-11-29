#!/usr/bin/env bash

set -e

# DATASETS=${DATASETS:-"*"}

SERVER_HOST=${SERVER_HOST:-"localhost"}

# SERVER_USERNAME=${SERVER_USERNAME:-"qdrant"}

SOURCE_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

function run_exp() {
    # sync 
    # sudo bash -c "echo 1 > /proc/sys/vm/drop_caches" 
    sudo rm -rf $SOURCE_DIR/results/*
    sudo rm -rf $SOURCE_DIR/engine/servers/milvus-single-node/volumes

    SERVER_PATH=$1
    ENGINE_NAME=$2
    DATASETS=$3
    MONITOR_PATH=$(echo "$ENGINE_NAME" | sed -e 's/[^A-Za-z0-9._-]/_/g')
    nohup bash -c "cd $SOURCE_DIR/monitoring && rm -f docker.stats.jsonl && bash monitor_docker.sh" > /dev/null 2>&1 &
    cd $SOURCE_DIR/engine/servers/$SERVER_PATH ; docker-compose down > /dev/null; docker-compose up -d > /dev/null
    sleep 30
    python3.11 $SOURCE_DIR/run.py --engines "$ENGINE_NAME" --datasets "${DATASETS}" --host "$SERVER_HOST" > /dev/null
    # exit
    cd $SOURCE_DIR/engine/servers/$SERVER_PATH ; docker-compose down > /dev/null
    cd $SOURCE_DIR/monitoring && mkdir -p results && sudo mv docker.stats.jsonl ./results/${MONITOR_PATH}-docker.stats.jsonl
}

function get_result() {
    res_file=`ls $SOURCE_DIR/results/ | grep -v 'upload'` 
    cat $SOURCE_DIR/results/$res_file | grep -E "mean_precisions|rps|p95_time" | awk '{print $2}' | sed 's#,##g'
}


SERVER_PATH=${1:-milvus-single-node}
ENGINE_NAME=${2:-milvus-p10}
DATASETS=${3:-glove-25-angular}

run_exp $SERVER_PATH $ENGINE_NAME $DATASETS
get_result


# "nlist": 32768, "m":5, "nbits":8
# "nprobe": 16384