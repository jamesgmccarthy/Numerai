from hpo import gbm_hpo, nn_hpo, ae_hpo
from dotenv import load_dotenv
import os
import numerapi


def credentials():
    dotenv_path = 'num_config.env'
    load_dotenv(dotenv_path=dotenv_path)
    pub_id = os.getenv('PUBLIC_ID')
    priv_key = os.getenv('PRIVATE_KEY')
    latest_round = os.getenv('LATEST_ROUND')
    return {'PUBLIC_ID': pub_id, 'PRIVATE_KEY': priv_key, 'LATEST_ROUND': latest_round}


def download_data(api: numerapi.NumerAPI, keys):
    if int(keys['LATEST_ROUND']) == api.get_current_round():
        return int(keys['LATEST_ROUND'])
    else:
        LATEST_ROUND = api.get_current_round()
        api.download_current_dataset('./data')
        return LATEST_ROUND


def update_env_file(env_vars):
    with open('num_config.env', 'w') as f:
        f.write(f'LATEST_ROUND={env_vars["LATEST_ROUND"]}\n')
        f.write(f'PUBLIC_ID={env_vars["PUBLIC_ID"]}\n')
        f.write(f'PRIVATE_KEY={env_vars["PRIVATE_KEY"]}\n')


def create_preds():
    pass


def main():
    keys = credentials()
    numapi = numerapi.NumerAPI(
        verbosity='INFO', public_id=keys['PUBLIC_ID'], secret_key=keys['PRIVATE_KEY'])
    keys['LATEST_ROUND'] = download_data(numapi, keys)
    update_env_file(keys)
    # gbm_hpo.main()
    # ae_hpo.main(embedding=False)
    # gbm_hpo.main(ae_train=True)
    # nn_hpo.main(train_ae=True)


if __name__ == '__main__':
    main()
