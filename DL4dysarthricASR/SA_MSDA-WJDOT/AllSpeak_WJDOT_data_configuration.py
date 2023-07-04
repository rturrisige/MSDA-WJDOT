import torch


def load_train_test_data(data_dir):
    """
    Parameters
    ----------
    data_dir: str
        path to data

    Returns
    ---------
    source_ey: torch array, [sources training samples, embedding dimension + n classes]
        concatenated source embeddings
    Ns_list: list of float
        list of number of source samples
    embedding_train: torch array, [target training samples, embedding dimension]
        target training embedding
    embedding_test: torch array,  [target testing samples, embedding dimension]
        target testing embedding
    y_train: torch array, [target training samples]
        target training embedding
    y_test: torch array,  [target testing samples]
        target testing embedding

    """
    # Load data
    source_ey = torch.load(data_dir + 'ey_all.pt')
    Ns_list = np.load(data_dir.split('exp')[0] + 'labelled_tr_samples.npy')
    embedding_train, y_train = torch.load(data_dir + 'Eu_Yu_train.pt')
    embedding_test, y_test = torch.load(data_dir + 'Eu_Yu_test.pt')
    return source_ey, embedding_train, y_train, embedding_test, y_test, Ns_list


def load_target_val_data(data_dir):
    """
    Parameters
    ----------
    data_dir: str
        path to data

    Returns
    ---------
    embedding_val: torch array, [target validation samples, embedding dimension]
        target validation embedding
    y_val: torch array,  [target validation samples]
        target validation embedding

    """
    # Load data
    embedding_val, y_val = torch.load(data_dir + 'Eu_Yu_val.pt')
    return embedding_val, y_val


def load_source_val_data(data_dir):
    """
    Parameters
    ----------
    data_dir: str
        path to data

    Returns
    ---------
    embedding_val: torch array, [source validation samples, embedding dimension]
        source validation embedding
    y_val: torch array,  [source validation samples]
        source validation embedding

    """
    # Load data
    embedding_val, y_val = torch.load(data_dir + 'labelled_val_EY.pt')
    return embedding_val, y_val