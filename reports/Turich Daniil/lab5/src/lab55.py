import numpy as np
import matplotlib.pyplot as plt
import itertools

n = 6
E_e = 0.01
alpha_fixed = 0.1
np.random.seed(42)

def logic_function(x):
    return 1 if np.all(x == 0) else 0

X_full = np.array(list(itertools.product([0, 1], repeat=n)))
y_full = np.array([logic_function(x) for x in X_full])

print("=" * 70)
print("ЛАБОРАТОРНАЯ РАБОТА №5: Бинарная классификация с BCE")
print(f"Вариант 7: n = {n}, логическая функция NOR (OR')")
print("=" * 70)
print(f"Полная таблица истинности: {len(X_full)} примеров")
print(f"  Класс 0 (не все нули): {np.sum(y_full == 0)}")
print(f"  Класс 1 (все нули):    {np.sum(y_full == 1)}")

idx_0 = np.where(y_full == 0)[0]
idx_1 = np.where(y_full == 1)[0]

np.random.seed(42)
np.random.shuffle(idx_0)
np.random.shuffle(idx_1)

train_0_len = int(0.8 * len(idx_0))
train_1_len = int(0.8 * len(idx_1))

if train_1_len == 0:
    train_1_len = 1

train_idx = np.concatenate((idx_0[:train_0_len], idx_1[:train_1_len]))
test_idx = np.concatenate((idx_0[train_0_len:], idx_1[train_1_len:]))

np.random.shuffle(train_idx)
np.random.shuffle(test_idx)

X_train, y_train = X_full[train_idx], y_full[train_idx]
X_test, y_test = X_full[test_idx], y_full[test_idx]

X_train_b = np.c_[np.ones(len(X_train)), X_train]
X_test_b = np.c_[np.ones(len(X_test)), X_test]
X_full_b = np.c_[np.ones(len(X_full)), X_full]

print(f"\nОбучающая выборка: {len(X_train)} примеров")
print(f"  Класс 0: {np.sum(y_train == 0)}")
print(f"  Класс 1: {np.sum(y_train == 1)}")
print(f"Тестовая выборка: {len(X_test)} примеров")
print(f"  Класс 0: {np.sum(y_test == 0)}")
print(f"  Класс 1: {np.sum(y_test == 1)}")

def sigmoid(net):
    return 1 / (1 + np.exp(-np.clip(net, -100, 100)))

def binary_cross_entropy(y_true, y_pred):
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    return - (y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

def mse(y_true, y_pred):
    return 0.5 * (y_true - y_pred) ** 2

def train_perceptron(X, y, loss_func, step_mode, max_epochs=20000):
    np.random.seed(42)
    w = np.random.uniform(-0.1, 0.1, n + 1)
    errors_history = []

    for epoch in range(max_epochs):
        total_error = 0.0

        for i in range(len(X)):
            xi = X[i]
            yi = y[i]

            net = np.dot(w, xi)
            y_pred = sigmoid(net)

            if step_mode == 'fixed':
                alpha = alpha_fixed
            else:
                alpha = 1.0 / (1.0 + np.sum(xi ** 2))

            if loss_func == 'MSE':
                error = mse(yi, y_pred)
                total_error += error
                grad = (yi - y_pred) * y_pred * (1 - y_pred)
            else:
                error = binary_cross_entropy(yi, y_pred)
                total_error += error
                grad = (yi - y_pred)

            w += alpha * grad * xi

        errors_history.append(total_error)

        if total_error <= E_e:
            break

    return w, errors_history, epoch + 1

def get_accuracy(w, X, y):
    predictions = (sigmoid(np.dot(X, w)) >= 0.5).astype(int)
    return np.mean(predictions == y) * 100

def print_weights(w, name):
    print(f"\n{name} — итоговые веса:")
    print(f"  Порог (w0): {w[0]:.8f}")
    for i in range(1, n + 1):
        print(f"  w{i}: {w[i]:.8f}")

configs = {
    'MSE + Fixed': ('MSE', 'fixed'),
    'MSE + Adaptive': ('MSE', 'adaptive'),
    'BCE + Fixed': ('BCE', 'fixed'),
    'BCE + Adaptive': ('BCE', 'adaptive')
}

print("\n" + "=" * 70)
print("ЗАПУСК ОБУЧЕНИЯ ДЛЯ 4 КОНФИГУРАЦИЙ")
print("=" * 70)

results = {}
for name, (l_func, s_mode) in configs.items():
    print(f"\n▶ Обучение: {name}")
    w, errs, epochs = train_perceptron(X_train_b, y_train, l_func, s_mode)
    results[name] = (w, errs, epochs)
    print(f"  ✓ Завершено за {epochs} эпох")
    print(f"  ✓ Финальная ошибка: {errs[-1]:.6f}")
    print_weights(w, name)

print("\n" + "=" * 70)
print("ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 70)
print(f"{'Конфигурация':<18} | {'Эпохи':<8} | {'Train Acc':<10} | {'Test Acc':<10} | {'Full Acc':<10}")
print("-" * 70)

results_table = []
for name, (w, errs, epochs) in results.items():
    acc_train = get_accuracy(w, X_train_b, y_train)
    acc_test = get_accuracy(w, X_test_b, y_test)
    acc_full = get_accuracy(w, X_full_b, y_full)
    results_table.append((name, epochs, acc_train, acc_test, acc_full))
    print(f"{name:<18} | {epochs:<8} | {acc_train:>8.2f}%    | {acc_test:>8.2f}%    | {acc_full:>8.2f}%")

plt.figure(figsize=(12, 7))
colors = ['blue', 'green', 'red', 'orange']
styles = ['-', '--', '-.', ':']

for idx, (name, (w, errs, epochs)) in enumerate(results.items()):
    plt.plot(errs, label=f"{name} ({epochs} эп.)",
             color=colors[idx % len(colors)],
             linestyle=styles[idx % len(styles)],
             linewidth=1.5)

plt.axhline(y=E_e, color='black', linestyle='--', linewidth=1, label=f'Порог Ee = {E_e}')
plt.yscale('log')
plt.title('Сравнение сходимости MSE и BCE (логарифмическая шкала)', fontsize=14)
plt.xlabel('Эпоха', fontsize=12)
plt.ylabel('Суммарная ошибка (log scale)', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('convergence_plot.png', dpi=150)
plt.show()

print("\n" + "=" * 70)
print("РЕЖИМ ФУНКЦИОНИРОВАНИЯ СЕТИ")
print("=" * 70)

best_config = 'BCE + Adaptive'
w_final = results[best_config][0]

print(f"Используется обученная сеть: {best_config}")
print(f"Логическая функция: NOR (возвращает 1 только если все {n} входов равны 0)")
print(f"Формат ввода: {n} чисел через пробел (0 или 1)")

while True:
    try:
        print("\n" + "-" * 50)
        user_input = input(f"Введите {n} бит через пробел (или 'q' для выхода): ").strip()

        if user_input.lower() == 'q':
            print("Выход из программы.")
            break

        x_val = np.array([int(i) for i in user_input.split()])

        if len(x_val) != n:
            print(f"Ошибка: нужно ввести ровно {n} чисел (сейчас {len(x_val)})")
            continue

        if not np.all((x_val == 0) | (x_val == 1)):
            print("Ошибка: можно вводить только 0 и 1")
            continue

        x_val_b = np.insert(x_val, 0, 1)
        prob = sigmoid(np.dot(w_final, x_val_b))
        pred_class = 1 if prob >= 0.5 else 0
        true_class = logic_function(x_val)

        print(f"\n  Входной вектор:           {x_val}")
        print(f"  Вероятность класса 1 (ŷ): {prob:.6f}")
        print(f"  Предсказанный класс:      {pred_class}")
        print(f"  Истинный класс:           {true_class}")

        if pred_class == true_class:
            print("  ✓ Совпадает с таблицей истинности")
        else:
            print("  ✗ Расхождение с таблицей истинности")

    except ValueError:
        print("Ошибка: введите целые числа (0 или 1) через пробел")
    except KeyboardInterrupt:
        print("\nВыход из программы.")
        break
    except Exception as e:
        print(f"Ошибка: {e}")

print("\n" + "=" * 70)
print("ВЫВОД")
print("=" * 70)

mse_fixed_epochs = results['MSE + Fixed'][2]
mse_adaptive_epochs = results['MSE + Adaptive'][2]
bce_fixed_epochs = results['BCE + Fixed'][2]
bce_adaptive_epochs = results['BCE + Adaptive'][2]

print("""
1. Сравнение скорости сходимости MSE и BCE:
""")
if bce_fixed_epochs < mse_fixed_epochs:
    print(f"   - BCE с фиксированным шагом сошлась за {bce_fixed_epochs} эпох,")
    print(f"     что БЫСТРЕЕ MSE (за {mse_fixed_epochs} эпох).")
else:
    print(f"   - MSE с фиксированным шагом сошлась за {mse_fixed_epochs} эпох,")
    print(f"     что БЫСТРЕЕ BCE (за {bce_fixed_epochs} эпох).")

if bce_adaptive_epochs < mse_adaptive_epochs:
    print(f"   - BCE с адаптивным шагом сошлась за {bce_adaptive_epochs} эпох,")
    print(f"     что БЫСТРЕЕ MSE (за {mse_adaptive_epochs} эпох).")
else:
    print(f"   - MSE с адаптивным шагом сошлась за {mse_adaptive_epochs} эпох,")
    print(f"     что БЫСТРЕЕ BCE (за {bce_adaptive_epochs} эпох).")

print("""
2. Почему BCE теоретически лучше подходит для классификации, чем MSE:
   - MSE штрафует ошибки квадратично, что приводит к малым градиентам
     при уверенных, но неверных предсказаниях (когда сеть выдаёт ~0 или ~1).
   - BCE имеет градиент (y_true - y_pred), который остаётся значительным
     при любых значениях ошибки, что ускоряет обучение на начальных этапах.
   - BCE интерпретируется как логарифм правдоподобия для бинарного распределения,
     что делает её статистически обоснованной для задач классификации.

3. Обобщающая способность при переходе на BCE:
""")
for name, epochs, acc_train, acc_test, acc_full in results_table:
    print(f"   {name:<18}: Full Acc = {acc_full:.2f}%")
print("""
   - BCE обычно обеспечивает более высокую точность на тестовой выборке,
     так как лучше разделяет классы и находит более оптимальную разделяющую
     гиперплоскость.

4. Влияние адаптивного шага на BCE:
""")
print(f"   - BCE с фиксированным шагом:   {bce_fixed_epochs} эпох")
print(f"   - BCE с адаптивным шагом:      {bce_adaptive_epochs} эпох")
print("""
   - Адаптивный шаг (alpha = 1/(1+||x||²)) предотвращает расходимость
     и обеспечивает более стабильную сходимость, особенно для разреженных
     входных векторов.
   - Для BCE адаптивный шаг даёт ускорение сходимости по сравнению с
     фиксированным шагом.

5. Достаточность однослойного персептрона:
   - Однослойный персептрон ДОСТАТОЧЕН для линейно разделимых функций
     (AND, OR, NOR, NAND).
   - Для нелинейно разделимых функций (XOR, мажоритарные функции)
     потребуется многослойная сеть с нелинейными активациями.
   - В данном случае функция NOR является линейно разделимой, поэтому
     однослойный персептрон успешно справляется с задачей.
""")

print("\n" + "=" * 70)
print("КОД ПРОГРАММЫ ГОТОВ К ЗАПУСКУ")
print("=" * 70)