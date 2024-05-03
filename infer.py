import argparse
import json
import os
import re
from typing import Callable, List

from paddlenlp import Taskflow
from tqdm import tqdm
import ipdb
from paddlenlp.utils.log import logger
logger.set_level('INFO')

class Processor:
    def __init__(self, 
                 select_strategy: str='all',
                 threshold: float=0.5,
                 selet_key: List[str] = ["text", "start", "end", "probability"],
                 ) -> None:
        
        self.select_strategy_fun = eval(f"self._{select_strategy}_postprocess")
        self.threshold = threshold if threshold else 0.5
        self.selet_key = selet_key if selet_key else ["text", "start", "end", "probability"]
    
    def _key_filter(strategy_fun):
        def select_key(self, each_entity_results):
            each_entity_results = strategy_fun(self, each_entity_results)
            for i, each_entity_result in enumerate(each_entity_results):
                each_entity_results[i] = {key: each_entity_result[key] for key in self.selet_key}
            return each_entity_results
            
    def postprocess(self, results):
        new_result = []
        for result in results:
            tmp = [{}]
            for entity in result[0]:
                tmp[0][entity] = self.select_strategy_fun(result[0][entity])
            new_result.append(tmp)
        return new_result
    
    # @_key_filter
    def _max_postprocess(self, each_entity_results):
        return [sorted(each_entity_results, key=lambda x: x['probability'], reverse=True)[0]]
        
    # @_key_filter
    def _threshold_postprocess(self, each_entity_results):
        return [entity for entity in each_entity_results if entity['probability'] >= self.threshold]
    
    # @_key_filter 
    def _all_postprocess(self, each_entity_results):
        return each_entity_results    
        
        
def inference(schema: List, 
              data_file_path: str,
              model: str = 'uie-base',
              task_path: str = None,
              postprocess_fun: Callable = lambda x: x):
    
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file {data_file_path} not found.")
    
    data = []
    logger.info(f"Loading data from {data_file_path}")
    with open(data_file_path, 'r', encoding='utf-8') as f:
        data = [{'context': line.strip()} for line in f]
    
    if task_path:
        uie = Taskflow(
            'information_extraction',
            schema=schema,
            task_path=task_path
        )
    else:
        uie = Taskflow(
            'information_extraction',
            schema=schema,
            model=model
        )
        
    
    result = [postprocess_fun([uie(context['context'])]) for context in tqdm(data[:2])]
    return result
    
if __name__ == '__main__':
    '''
    uie_result_example = [{'法院': [{'text': '士林地方法院', 'start': 2, 'end': 8, 'probability': 0.6794323139415823}], 
                           '原告': [{'text': '張益銘', 'start': 27, 'end': 30, 'probability': 0.8313725342392786}, 
                                  {'text': '張員', 'start': 4103, 'end': 4105, 'probability': 0.6192862186953363}], 
                           '被告': [{'text': '張員', 'start': 4103, 'end': 4105, 'probability': 0.2760115535638761}]}]
    '''
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_file_path', default="./data/test.txt", type=str)
    parser.add_argument('--result_save_dir', default="./data/result", type=str)
    parser.add_argument('--task_path', default=None, type=str)
    parser.add_argument('--schema', default=None, type=str)
    
    args = parser.parse_args()
    schema = ['法院', '原告', '被告']
    
    uie_processor = Processor(select_strategy='max', 
                              threshold=0.5)
    
    result = inference(data_file_path=args.data_file_path, 
              task_path=args.task_path, 
            #   schema=args.schema
            schema=schema,
            postprocess_fun=uie_processor.postprocess
              )
    
    if not os.path.exists(args.result_save_dir):
        logger.info(f"Create result save dir {args.result_save_dir}")
        os.makedirs(args.result_save_dir)
        
    with open(args.result_save_dir + '/result.json', 'w', encoding='utf-8') as f:
        jsonString = json.dumps(result, ensure_ascii=False)
        f.write(jsonString)
    logger.info(f"Save result to {args.result_save_dir}/result.json")