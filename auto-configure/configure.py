import json
import yaml

with open('/home/ytn/milvusTuning/auto-configure/index_param.json', 'r') as f:
    INDEX_PARAM_DICT = json.load(f)

CONF_PATH = r'/home/ytn/milvusTuning/vector-db-benchmark-master/experiments/configurations/milvus-single-node.json'
ORIGIN_PATH = r'/home/ytn/milvusTuning/vector-db-benchmark-master/engine/servers/milvus-single-node/milvus.yaml.backup'
ADJUST_PATH = r'/home/ytn/milvusTuning/vector-db-benchmark-master/engine/servers/milvus-single-node/milvus.yaml'

def filter_index_rule(conf):
    for item in INDEX_PARAM_DICT.keys():
        if item not in list(conf.keys()):
            conf[item] = INDEX_PARAM_DICT[item]['default']

    # print(conf)
    conf['nprobe'] = int(conf['nlist'] * conf['nprobe'] / 100)
    conf['nprobe'] = max(1, conf['nprobe'])
    # if conf['nprobe'] > conf['nlist']:
    #     conf['nprobe'] = conf['nlist']

    if conf['index_type'] in ['AUTOINDEX', 'FLAT']:
        building_params = {}
        searching_params = {}
    elif conf['index_type'] in ['IVF_FLAT', 'IVF_SQ8']:
        building_params = {'nlist': conf['nlist']}
        searching_params = {'nprobe': conf['nprobe']}
    elif conf['index_type'] in ['IVF_PQ']:
        building_params = {'nlist': conf['nlist'], 'm': conf['m'], 'nbits': conf['nbits']}
        searching_params = {'nprobe': conf['nprobe']}
    elif conf['index_type'] in ['HNSW']:
        building_params = {'M': conf['M'], 'efConstruction': conf['efConstruction']}
        searching_params = {'ef': conf['ef']}
    elif conf['index_type'] in ['SCANN']:
        building_params = {'nlist': conf['nlist']}
        searching_params = {'nprobe': conf['nprobe'], 'reorder_k': conf['reorder_k']}

    return conf['index_type'], building_params, searching_params

def configure_index(index_type, building_params, searching_params):
    conf_path = CONF_PATH
    with open(conf_path, 'r') as f:
        conf = json.load(f)
    conf[0]['upload_params']['index_type'] = index_type
    conf[0]['upload_params']['index_params'] = building_params
    conf[0]['search_params'][0]['params'] = searching_params
    with open(conf_path, 'w') as f:
        f.write(json.dumps(conf, indent=2))


def filter_system_rule(conf):
    for k in conf.keys():
        if k in ['dataCoord*segment*sealProportion']:
            conf[k] = conf[k] / 100
    return conf


def configure_system(params):
    origin_path = ORIGIN_PATH
    adjust_path = ADJUST_PATH
    with open(origin_path, 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    for k,v in params.items():
        pos = k.split('*')
        if len(pos) == 4:
            conf[pos[0]][pos[1]][pos[2]][pos[3]] = v
        elif len(pos) == 3:
            conf[pos[0]][pos[1]][pos[2]] = v
        elif len(pos) == 2:
            conf[pos[0]][pos[1]] = v
    with open(adjust_path, 'w') as f:
        yaml.dump(conf, f)


if __name__ == '__main__':
    # configure_index(*filter_index_rule({'index_type':'AUTOINDEX'}))
    configure_index(*filter_index_rule({'index_type':'SCANN', 'nlist':1014, 'nprobe':17, 'reorder_k': 716}))

    configure_system(filter_system_rule({}))
    # configure_system(filter_system_rule(
    # {
    #     'dataCoord*segment*maxSize': 100, 
    #     'dataCoord*segment*sealProportion': 1, 
    #     'queryCoord*autoHandoff': True,
    #     'queryCoord*autoBalance': False, 
    #     'common*gracefulTime': 55093, 
    #     'dataNode*segment*insertBufSize': 1000, 
    #     'rootCoord*minSegmentSizeToEnableIndex': 4960
    # }))