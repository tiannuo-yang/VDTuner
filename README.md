<div  align="center">    
<img src="./logo2.png" alt="VDTuner" width="60%">
</div>

---

<!-- # VDTuner: Towards High Search Speed and Recall Rate by Auto-Configuring Vector Data Management System -->

VDTuner is tested on a server configured with CentOS 7.9.2009 (Linux 5.5.0) and Python 3.11. The evaluated Vector Database Management System and benchamark is Mivlus (version 2.3.1) and vector-db-benchmark.

## Dependencies
1. Make sure [Milvus (2.3.1 version with docker-compose)](https://milvus.io/docs/install_standalone-docker.md) is deployed on your server.
2. Install benchmark [vector-db-benchmark](https://github.com/qdrant/vector-db-benchmark) from github source.
3. Install Python 3.11 and necessary package [BoTorch](https://botorch.org/docs/getting_started).
   
## Preparations
#### Modify the defualt engine in benchmark.  
- According to [Milvus configuration document](https://milvus.io/docs/configure-docker.md), modify (or add) the file `docker-compose.yml` (Milvus startup file), `milvus.yaml` (Milvus configuration file for utilizing) and `milvus.yaml.backup` (Milvus configuration file for copying and modifying) in your benchmark path `vector-db-benchmark-master/engine/servers/milvus-single-node`.  
- Copy the modified `run_engine.sh` to `vector-db-benchmark-master/run_engine.sh`.  
- After that, you can test if Milvus deployed successfully on your server by testing a relatively small dataset. Go to `vector-db-benchmark-master` and run:
     `sudo ./run_engine.sh "" "" random-100`.
  
#### Download dataset.  
- Take [GloVe](http://ann-benchmarks.com/glove-100-angular.hdf5) as an example, download it to `vector-db-benchmark-master/datasets/glove-100-angular/glove-100-angular.hdf5`.

#### Specify the similarity search requests.  
- Modify the file `vector-db-benchmark-master/experiments/configurations/milvus-single-node.json` to a defualt index configuration as follow. The parameter `parallel` can be modified according to your server specifications.  
   ```json
   [
     {
       "name": "milvus-p10",
       "engine": "milvus",
       "connection_params": {},
       "collection_params": {},
       "search_params": [
         {
           "parallel": 10,
           "params": {}
         }
       ],
       "upload_params": {
         "parallel": 10,
         "index_type": "AUTOINDEX",
         "index_params": {}
       }
     }
   ]
   ```

- Specify your dataset and timeout limit. In file `auto-configure/vdtuner/utils.py` (line 117), assume we test dataset GloVe, with a maximum of 15 minutes for each workload replay:
   ```python
   result = sp.run(f'sudo timeout 900 {RUN_ENGINE_PATH} "" "" glove-100-angular', shell=True, stdout=sp.PIPE)
   ```

#### Specify your config file path.  
- To run VDTuner, you need to specify the configuration file of tuning parameters and benchmark path. Here is an example.  
  In file `auto-configure/configure.py`, line 4-9:
  ```python
   with open('/home/ytn/milvusTuning/auto-configure/index_param.json', 'r') as f:
       INDEX_PARAM_DICT = json.load(f)
   
   CONF_PATH = r'/home/ytn/milvusTuning/vector-db-benchmark-master/experiments/configurations/milvus-single-node.json'
   ORIGIN_PATH = r'/home/ytn/milvusTuning/vector-db-benchmark-master/engine/servers/milvus-single-node/milvus.yaml.backup'
   ADJUST_PATH = r'/home/ytn/milvusTuning/vector-db-benchmark-master/engine/servers/milvus-single-node/milvus.yaml'
  ```

  In file `auto-configure/vdtuner/utils.py`, line 13-14:
  ```python
   KNOB_PATH = r'/home/ytn/milvusTuning/auto-configure/whole_param.json'
   RUN_ENGINE_PATH = r'/home/ytn/milvusTuning/vector-db-benchmark-master/run_engine.sh'
  ```

## Run VDTuner
### Start auto-tuning
- Congratulations! You can run VDTuner to optimize your vector database now! Go to `auto-configure/vdtuner/` and run:
  ```shell
   python3.11 main_tuner.py
  ```
  Note that this would take very long time (about 30000s for 200 iterations with dataset GloVe), because VDTuner iteratively performs workload replay and configuration recommendation. You can change the number of iterations as desired in `main_tuner.py`.
- Results of sampled configurations and optimizer's internal information will be logged to `record.log` and `pobo_record.log` in real time. Here is an example output in `record.log`:
  ```
   [1] 125 {index_type: FLAT, nlist: 128, nprobe: 10, m: 10, nbits: 8, M: 32, efConstruction: 256, ef: 500, reorder_k: 500} {dataCoord*segment*maxSize: 512, dataCoord*segment*sealProportion: 0.23, queryCoord*autoHandoff: True, queryCoord*autoBalance: True, common*gracefulTime: 5000, dataNode*segment*insertBufSize: 16777216, rootCoord*minSegmentSizeToEnableIndex: 1024} 230.5802223315391 0.9999830000000002 125
   [2] 214 {index_type: IVF_FLAT, nlist: 128, nprobe: 10, m: 10, nbits: 8, M: 32, efConstruction: 256, ef: 500, reorder_k: 500} {dataCoord*segment*maxSize: 512, dataCoord*segment*sealProportion: 0.23, queryCoord*autoHandoff: True, queryCoord*autoBalance: True, common*gracefulTime: 5000, dataNode*segment*insertBufSize: 16777216, rootCoord*minSegmentSizeToEnableIndex: 1024} 1086.9213657571365 0.8496440000000001 88
   [3] 302 {index_type: IVF_SQ8, nlist: 128, nprobe: 10, m: 10, nbits: 8, M: 32, efConstruction: 256, ef: 500, reorder_k: 500} {dataCoord*segment*maxSize: 512, dataCoord*segment*sealProportion: 0.23, queryCoord*autoHandoff: True, queryCoord*autoBalance: True, common*gracefulTime: 5000, dataNode*segment*insertBufSize: 16777216, rootCoord*minSegmentSizeToEnableIndex: 1024} 908.6127610863159 0.8461550000000001 88
  ...
  ```

## Citation
If you use VDTuner in your scientific article, please cite our ICDE 2024 paper:  
```
@inproceedings{yang2024vdtuner,  
     title={VDTuner: Automated Performance Tuning for Vector Data Management Systems},  
     author={Yang, Tiannuo and Hu, Wen and Peng, Wangqi and Li, Yusen and Li, Jianguo and Wang, Gang and Liu, Xiaoguang},  
     booktitle={2024 IEEE 40th International Conference on Data Engineering (ICDE)},  
     year={2024}  
}
```

## Contributors
[Tiannuo Yang](https://github.com/tiannuo-yang) <yangtn@nbjl.nankai.edu.cn>  
[Wangqi Peng](https://github.com/yanxiaoqi932) <pengwq@nbjl.nankai.edu.cn>  
-- From Lab [NBJL](https://nbjl.nankai.edu.cn/) and [Ant Group](https://www.antgroup.com/)
