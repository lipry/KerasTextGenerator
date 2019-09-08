import matplotlib.pyplot as plt


def plot_train_validation_loss(loss, val_loss, save_file="./logs/train_val_loss.png"):
    # Plot training & validation loss values
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig(save_file)