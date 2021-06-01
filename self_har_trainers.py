import tensorflow as tf
import os

__author__ = "C. I. Tang"
__copyright__ = "Copyright (C) 2021 C. I. Tang"

"""
Complementing the work of Tang et al.: SelfHAR: Improving Human Activity Recognition through Self-training with Unlabeled Data
@article{tang2021selfhar,
  title={SelfHAR: Improving Human Activity Recognition through Self-training with Unlabeled Data},
  author={Tang, Chi Ian and Perez-Pozuelo, Ignacio and Spathis, Dimitris and Brage, Soren and Wareham, Nick and Mascolo, Cecilia},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume={5},
  number={1},
  pages={1--30},
  year={2021},
  publisher={ACM New York, NY, USA}
}

Access to Article:
    https://doi.org/10.1145/3448112
    https://dl.acm.org/doi/abs/10.1145/3448112

Contact: cit27@cl.cam.ac.uk
"""

def composite_train_model(
    full_model, 
    training_set,
    working_directory, 
    callbacks, 
    epochs,
    validation_set=None,
    batch_size=200,
    tag="har", 
    use_tensor_board_logging=True,
    steps_per_epoch=None,
    verbose=1
):
    
    best_model_file_name = os.path.join(working_directory, "models", f"{tag}_best.hdf5")
    last_model_file_name = os.path.join(working_directory, "models", f"{tag}_last.hdf5")
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(
        best_model_file_name,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_weights_only=False,
        verbose=verbose
    )

    local_callbacks = [best_model_callback]

    if use_tensor_board_logging:
        logdir = os.path.join(working_directory, "logs", tag)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)
        local_callbacks.append(tensorboard_callback)

    training_history = full_model.fit(
        x=training_set[0],
        y=training_set[1],
        validation_data=(validation_set[0], validation_set[1]),

        batch_size=batch_size,
        shuffle=True,
        epochs=epochs,
        callbacks = callbacks + local_callbacks,

        steps_per_epoch=steps_per_epoch,
        verbose=verbose
    )

    
    full_model.save(last_model_file_name)
    
    return best_model_file_name, last_model_file_name



