import matplotlib.pyplot as plt
import numpy as np

# --- 1. Подготовка данных ---
sizes = [5, 1000000, 50000000]
configs = ["1,32", "1,128", "32,32", "64,128", "256,256", "512,512", "1024,1024"]

cpu_times = [0.001827, 6.432408, 259.429866]

gpu_data = {
    5: [2.637824, 1.257472, 0.913120, 1.061888, 0.939808, 1.002496, 0.910336],
    1000000: [12.558656, 3.920352, 1.314944, 1.127040, 1.131488, 0.980960, 1.557504],
    50000000: [524.310669, 150.795197, 20.657536, 6.348288, 5.617248, 5.848064, 6.059264]
}

# Находим лучшие времена GPU для каждого размера для сравнения с CPU
best_gpu_times = [min(gpu_data[s]) for s in sizes]
speedup = [cpu / gpu for cpu, gpu in zip(cpu_times, best_gpu_times)]

# --- 2. Визуализация ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(wspace=0.3)

# График 1: Эффективность конфигураций GPU (для больших и средних данных)
for s in sizes[1:]:  # Маленькие данные (5) не берем, там только шум и оверхед
    axes[0].plot(configs, gpu_data[s], marker='o', label=f'Size: {s}')

axes[0].set_yscale('log')
axes[0].set_title("Эффективность конфигураций GPU", fontsize=12)
axes[0].set_xlabel("Конфигурация <<<Grid, Block>>>")
axes[0].set_ylabel("Время выполнения (ms) - log scale")
axes[0].grid(True, which="both", ls="-", alpha=0.5)
axes[0].legend()

# График 2: Сравнение CPU vs GPU (Best case)
x = np.arange(len(sizes))
width = 0.35
axes[1].bar(x - width/2, cpu_times, width, label='CPU', color='tomato')
axes[1].bar(x + width/2, best_gpu_times, width, label='GPU (Best Config)', color='skyblue')

axes[1].set_yscale('log')
axes[1].set_xticks(x)
axes[1].set_xticklabels([str(s) for s in sizes])
axes[1].set_title("Сравнение CPU vs GPU", fontsize=12)
axes[1].set_xlabel("Количество элементов")
axes[1].set_ylabel("Время выполнения (ms) - log scale")
axes[1].legend()
axes[1].grid(axis='y', ls="--", alpha=0.7)

# График 3: Ускорение (Speedup)
axes[2].plot([str(s) for s in sizes], speedup, marker='s', color='green', linewidth=2)
axes[2].set_title("Ускорение (Speedup = T_cpu / T_gpu)", fontsize=12)
axes[2].set_xlabel("Количество элементов")
axes[2].set_ylabel("Кратность ускорения")
axes[2].grid(True, ls="-", alpha=0.5)
# Добавим горизонтальную линию на уровне 1 (где CPU = GPU)
axes[2].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='CPU = GPU')

plt.tight_layout()
plt.show()

# Печать таблицы ускорения для отчета
print(f"{'Size':>10} | {'Speedup':>10}")
print("-" * 23)
for s, sp in zip(sizes, speedup):
    print(f"{s:>10} | {sp:>10.2f}x")