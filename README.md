# VDTuner: Towards High Search Speed and Recall Rate by Auto-Configuring Vector Data Management System

VDTuner is tested on a server configured with CentOS 7.9.2009 (Linux 5.5.0) and Python 3.11. The ecaluated Vector Database Management System and benchamark is Mivlus (version 2.3.1) and vector-db-benchmark.

### Dependencies
1. Make sure [Milvus (2.3.1 version with docker-compose)](https://milvus.io/docs/install_standalone-docker.md) is deployed on your server.
2. Install benchmark [vector-db-benchmark](https://github.com/qdrant/vector-db-benchmark) from github source.
3. Install Python 3.11 alone with the following packages:
   
### Preparations
1. Modify the defualt engine in benchmark.
   According to [Milvus configuration document], modify (or add) the file `docker-compose.yml` (Milvus startup file), `milvus.yaml` (Milvus configuration file for utilizing) and `milvus.yaml.backup` (Milvus configuration file for copying and modifying) in your benchmark path `vector-db-benchmark-master/engine/servers/milvus-single-node`.
