'''
run mulitple pipelines for quick comparison

NOTE on the two functions here. run_multiple_pipelines used to hardcode the transform onto
the TEST side only while run_multiple_pipelines_diff applied it to both, which is the entire
reason the two families of difference-in-difference figures came out an order of magnitude
apart on the same axis label (FINDINGS.md 3.3). transform_target is now explicit in both.

Both also used to hand target and test the same seed, which makes get_indicies return
IDENTICAL images -- the KL then only reflects the noise draw. Pass partition='ab' to draw
target and test from guaranteed-disjoint halves instead.
'''
from tqdm import tqdm
from h_test_IQM.pipeline import get_scores


def run_multiple_pipelines(pipeline_list, cache_data=True, scorer='entropy-2-mse'):
    # run the tests for each configuration
    test_results = []
    preloaded_ims = {}
    for run in pipeline_list:
        test_result = []
        if 'name' not in run:
            run['name'] = run['dataset']
        partition = run.get('partition', None)
        for i in tqdm(range(run['runs']), desc=run['name']):
            outputs = get_scores(
                dataset_target=run['dataset'],
                dataset_test=run['dataset'],
                target_labels=run['data_labels'],
                test_labels=run['data_labels'],
                transform_target=run.get('transform_target', None),
                transform_test=run.get('transform_test', run.get('noise', None)),
                scorer=run.get('scorer', scorer),
                test=run['test'],
                dev=run['dev'],
                dataset_proportion_target=run['dataset_proportion'],
                dataset_proportion_test=run['dataset_proportion'],
                partition_target='a' if partition == 'ab' else None,
                partition_test='b' if partition == 'ab' else None,
                seed=i,
                _print=False,
                preloaded_ims=preloaded_ims,
            )
            test_result.append(outputs['results'][run['test']])
            if cache_data == True:
                preloaded_ims = outputs['preloaded_ims']
        test_results.append(test_result)
    return test_results


def run_multiple_pipelines_diff(pipeline_list, cache_data=True):
    '''
    For use with different test and target datasets/ transforms
    '''
    # run the tests for each configuration
    test_results = []
    preloaded_ims = {}
    for run in pipeline_list:
        test_result = []
        if 'name' not in run:
            run['name'] = f"{run['dataset_test']} - {run['dataset_target']}"
        for i in tqdm(range(run['runs']), desc=run['name']):
            outputs = get_scores(
                dataset_target=run['dataset_target'],
                dataset_test=run['dataset_test'],
                target_labels=run['target_labels'],
                test_labels=run['test_labels'],
                transform_target=run['transform_target'],
                transform_test=run['transform_test'],
                scorer='entropy-2-mse',
                test=run['test'],
                dev=run['dev'],
                dataset_proportion_target=run['dataset_proportion_target'],
                dataset_proportion_test=run['dataset_proportion_test'],
                seed=i,
                _print=False,
                preloaded_ims=preloaded_ims,
            )
            test_result.append(outputs['results'][run['test']])
            if cache_data == True:
                preloaded_ims = outputs['preloaded_ims']
        test_results.append(test_result)
    return test_results
