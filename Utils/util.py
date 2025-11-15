import anyconfig


def parse_config(config: dict) -> dict:
    """根据yaml文件解析的字典进行合并，如果有base字段的话，则进行加载base字段。
    用法：
    config = anyconfig.load('../../Configs/icdar2015_resnet18_FPN_DBhead_polyLR_all.yaml')
    config = parse_config(config)
    """
    base_file_list = config.pop('base')
    base_config = {}
    for base_file in base_file_list:  
        tmp_config = anyconfig.load(open(base_file, 'rb'))
        if 'base' in tmp_config:
            tmp_config = parse_config(tmp_config)
        anyconfig.merge(tmp_config, base_config)
        base_config = tmp_config
    anyconfig.merge(base_config, config)  
    return base_config
