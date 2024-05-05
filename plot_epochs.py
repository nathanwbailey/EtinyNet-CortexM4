import matplotlib.pyplot as plt
acc_to_plot_val = []
acc_to_plot_train = []

acc_to_plot_val_5 = []
acc_to_plot_train_5 = []
with open("training_student_teacher_3.txt", "r") as plot_file:
    for line in plot_file:
        line = line.rstrip()
        if 'val_output_1_Top-1 Accuracy' in line:
            s_idx = line.index('val_output_1_Top-1 Accuracy')
            acc = float(line[s_idx+29:s_idx+35])
            acc_to_plot_val.append(acc)
        if 'output_1_Top-1 Accuracy' in line:
            s_idx = line.index('output_1_Top-1 Accuracy')
            acc = float(line[s_idx+26:s_idx+32])

            acc_to_plot_train.append(acc)
        if 'val_output_1_Top-5 Accuracy' in line:
            s_idx = line.index('val_output_1_Top-5 Accuracy')
            acc = float(line[s_idx+29:s_idx+35])
            acc_to_plot_val_5.append(acc)
        if 'output_1_Top-5 Accuracy' in line:
            s_idx = line.index('output_1_Top-5 Accuracy')
            acc = float(line[s_idx+26:s_idx+32])
            acc_to_plot_train_5.append(acc)


plt.plot(acc_to_plot_val, color='m', label='Top-1 Validation Accuracy')
plt.plot(acc_to_plot_train, color='b', label='Top-1 Training Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('top-1_Accuracy_3.png')

plt.clf()
plt.plot(acc_to_plot_val_5, color='m', label='Top-5 Validation Accuracy')
plt.plot(acc_to_plot_train_5, color='b', label='Top-5 Training Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('top-5_Accuracy_3.png')
