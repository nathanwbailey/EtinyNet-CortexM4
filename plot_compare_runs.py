import matplotlib.pyplot as plt
acc_to_plot_1 = []
acc_to_plot_2 = []
acc_to_plot_3 = []


with open("training_student_teacher_3.txt", "r") as plot_file:
    for line in plot_file:
        line = line.rstrip()
        if 'val_output_1_Top-5 Accuracy' in line:
            s_idx = line.index('val_output_1_Top-5 Accuracy')
            acc = float(line[s_idx+29:s_idx+35])
            acc_to_plot_1.append(acc)

with open("training_student_teacher_30.txt", "r") as plot_file:
    for line in plot_file:
        line = line.rstrip()
        if 'val_output_1_Top-5 Accuracy' in line:
            s_idx = line.index('val_output_1_Top-5 Accuracy')
            acc = float(line[s_idx+29:s_idx+35])
            acc_to_plot_2.append(acc)

with open("Training_accuracy_increase.txt", "r") as plot_file:
    for line in plot_file:
        line = line.rstrip()
        if 'val_Top-5 Accuracy' in line:
            s_idx = line.index('val_Top-5 Accuracy')
            acc = float(line[s_idx+20:s_idx+26])
            acc_to_plot_3.append(acc)


plt.plot(acc_to_plot_1, color='m', label='Student-Teacher Val Top-5 Accuracy, Huber loss weight=3')
plt.plot(acc_to_plot_2, color='b', label='Student-Teacher Val Top-5 Accuracy, Huber loss weight=30')
plt.plot(acc_to_plot_3, color='g', label='Original Val Top-5 Accuracy')


plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('top-5_accuracy_compare.png')


