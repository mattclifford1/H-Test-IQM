'''
run mulitple pipelines for quick comparison
'''
from tqdm import tqdm
from h_test_IQM.pipeline import get_scores


def run_multiple_pipelines(pipeline_list, cache_data=True):
    # run the tests for each configuration
    test_results = []
    preloaded_ims = {}
    for run in pipeline_list:
        test_result = []
        if 'name' not in run:
            run['name'] = run['dataset']
        for i in tqdm(range(run['runs']), desc=run['name']):
            outputs = get_scores(
                dataset_target=run['dataset'],
                dataset_test=run['dataset'],
                target_labels=run['data_labels'],
                test_labels=run['data_labels'],
                transform_test=run['noise'],
                scorer='entropy-2-mse',
                test=run['test'],
                dev=run['dev'],
                dataset_proportion_target=run['dataset_proportion'],
                dataset_proportion_test=run['dataset_proportion'],
                seed=i,
                _print=False,
                preloaded_ims=preloaded_ims,
            )
            test_result.append(outputs['results'][run['test']])
            if cache_data == True:
                preloaded_ims = outputs['preloaded_ims']
        test_results.append(test_result)
    return test_results