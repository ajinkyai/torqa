import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set(font_scale=1.5)


# objects = ('1', '2', '3', '4', '5')
# y_pos = np.arange(len(objects))
# performance = [52.2, 90.9, 91.7, 90.2, 90.8]
#
# # plt.bar(y_pos, performance, align='center', alpha=0.5)
# sns.barplot(y_pos, performance, palette='BuGn_d')
# plt.xticks(y_pos, objects)
# plt.ylim((50, 100))
# plt.xlabel('Hops')
# plt.ylabel('Accuracy')
# plt.title('Accuracy as a function of Hops')
# plt.tight_layout()
# plt.show()



save_params = pickle.load(open("plotdata/save_params.p", "rb"))
epoch_test_acc, epoch_train_acc, epoch_loss, epoch_lr =  save_params

a = 1


# plt.title('Training Loss v/s Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Training Loss')
# epochs = list(epoch_loss.keys())
# plt.plot(epochs, list(epoch_loss.values()), linewidth=2, label="Training loss")
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()


#
# plt.title('Test accuracy v/s Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Test accuracy')
# plt.ylim((45, 100))
# testloss = list(map(lambda x: x*100, list(epoch_test_acc.values())))
# epochs = list(epoch_test_acc.keys())
# plt.plot(epochs, testloss, linewidth=2, marker='o', label="Test accuracy")
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()


objects = ('Without GC', 'With GC')
y_pos = np.arange(len(objects))
performance = [90, 91.7]

# plt.bar(y_pos, performance, align='center', alpha=0.5)
sns.barplot(y_pos, performance, palette='BuGn_d')
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Comparison of model with and without GC')
plt.tight_layout()
plt.show()