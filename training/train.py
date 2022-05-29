# -*- coding: utf-8 -*-
# 💾⚙️🔮

__author__ = "Daulet N."
__email__ = "daulet.nurmanbetov@gmail.com"

from simpletransformers.ner import NERModel

VALID_LABELS = ['OU', 'OO', '.O', '!O', ',O', '.U', '!U', ',U', ':O', ';O', ':U', "'O", '-O', '?O', '?U']

def e2e_train():
    steps, tr_details = train_model()
    print(f"Steps: {steps}; Train details: {tr_details}")


def train_model():
    """
    Trains simpletransformers model
    """
    print('Start training')
    # Create a NERModel
    model = NERModel("bert", "dbmdz/bert-base-german-uncased",
                     args={"overwrite_output_dir": True,
                           "num_train_epochs": 3,
                           "max_seq_length": 512,
                           "lazy_loading": True},
                     labels=VALID_LABELS)

    # # Train the model
    steps, tr_details = model.train_model('rpunct_train_set.txt')
    return steps, tr_details


if __name__ == "__main__":
    print("Training the model.")
    e2e_train()
