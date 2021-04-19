import data_loading.utils as utils
import os

def check_test_files(root_dir: str = './data'):
    data_path = utils.get_data_path(root_dir=root_dir)
    test_files_path = f'{data_path}/test_files'
    # TODO Check to make sure all era's present
    if os.path.isdir(test_files_path):
        return test_files_path, True
    else:
        os.makedirs(test_files_path)
        df = utils.load_data(root_dir=root_dir, mode='test')
        df['era'][df['era'] == 'eraX'] = 'era999'
        for era in df['era'].unique():
            path = f'{test_files_path}/{era}'
            df[df['era'] == era].to_csv(f'{path}.csv')
        return test_files_path, True


def create_predictions(root_dir: str = './data', models: dict = {}):
    test_files_path, test_files_exist = check_test_files(root_dir)
    if 
