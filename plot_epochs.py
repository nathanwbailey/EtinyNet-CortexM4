import matplotlib.pyplot as plt
acc_to_plot_val = []
acc_to_plot_train = []

acc_to_plot_val_5 = []
acc_to_plot_train_5 = []
with open("Training_accuracy_increase.txt", "r") as plot_file:
    for line in plot_file:
        line = line.rstrip()
        if 'val_Top-1 Accuracy' in line:
            s_idx = line.index('val_Top-1 Accuracy')
            acc = float(line[s_idx+20:s_idx+26])
            acc_to_plot_val.append(acc)
        if 'Top-1 Accuracy' in line:
            s_idx = line.index('Top-1 Accuracy')
            acc = float(line[s_idx+17:s_idx+23])
            acc_to_plot_train.append(acc)
        if 'val_Top-5 Accuracy' in line:
            s_idx = line.index('val_Top-5 Accuracy')
            acc = float(line[s_idx+20:s_idx+26])
            acc_to_plot_val_5.append(acc)
        if 'Top-5 Accuracy' in line:
            s_idx = line.index('Top-5 Accuracy')
            acc = float(line[s_idx+17:s_idx+23])
            acc_to_plot_train_5.append(acc)


plt.plot(acc_to_plot_val, color='m', label='Top-1 Validation Accuracy')
plt.plot(acc_to_plot_train, color='b', label='Top-1 Training Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('top-1_Accuracy.png')

plt.clf()
plt.plot(acc_to_plot_val_5, color='m', label='Top-5 Validation Accuracy')
plt.plot(acc_to_plot_train_5, color='b', label='Top-5 Training Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('top-5_Accuracy.png')
