import torch as th
import pickle as pkl


def compute_accuracy(model, headlines, stances, bodies):
    model.eval()
    with th.no_grad():
        outputs = model.forward(headlines, bodies).argmax(axis=1)
        accuracy = (outputs == stances).sum().float() / stances.numel()
        return accuracy

    
def load_vocab_dict(path):
    with open(path, 'rb') as vocab_file:
        vocab_dict = pkl.load(vocab_file)
        
    return vocab_dict


def save_vocab_dict(path, vocab_dict):
    with open(path, 'wb') as vocab_file:
        pkl.dump(vocab_dict, vocab_file)


def load_model_weights(model_weights_path, model, device=th.device('cpu')):
    model.load_state_dict(th.load(model_weights_ath, map_location=device))
    model.eval()
    
    return model


def save_model_weights(model_weights_path, model):
    th.save(model.state_dict(), model_weights_path)
    
    
def save_preprocessed_data(path_to_data, data):
    (headlines_train, stances_train, bodies_train) = data['train']
    (headlines_dev, stances_dev, bodies_dev) = data['dev']
    
    with open(path_to_data, 'wb') as data_file:
        pkl.dump((
            headlines_train,
            headlines_dev,
            stances_train,
            stances_dev,
            bodies_train,
            bodies_dev
        ), data_file)


def load_preprocessed_data(path_to_data):    
    with open(path_to_data, 'rb') as data_file:
        (
            headlines_train,
            headlines_dev,
            stances_train,
            stances_dev,
            bodies_train,
            bodies_dev
        ) = pkl.load(data_file)
    
    return {
        'train': (headlines_train, stances_train, bodies_train),
        'dev': (headlines_dev, stances_dev, bodies_dev)
    }