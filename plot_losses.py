import matplotlib.pyplot as plt
loss_to_plot_1 = []
loss_to_plot_2 = []


with open("training_student_teacher_30.txt", "r") as plot_file:
    for line in plot_file:
        line = line.rstrip()
        if 'val_output_1_loss' in line:
            s_idx = line.index('val_output_1_loss')
            loss = float(line[s_idx+19:s_idx+25])
            loss_to_plot_1.append(loss)
        if 'val_output_2_loss' in line:
            s_idx = line.index('val_output_2_loss')
            loss = float(line[s_idx+19:s_idx+25])
            loss_to_plot_2.append(loss)

plt.plot(loss_to_plot_1, color='m', label='Cross-Entropy Validation Loss')


plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig('val_cross_entropy_loss_30.png')

plt.clf()
plt.plot(loss_to_plot_2, color='b', label='Huber Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig('val_huber_loss_30.png')

