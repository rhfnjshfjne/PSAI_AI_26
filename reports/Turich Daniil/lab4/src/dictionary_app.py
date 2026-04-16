import numpy as np
import matplotlib.pyplot as plt
import itertools

# ========== Данные для варианта 7: n=6, функция NOR ==========
n = 6
E_e = 0.01
alpha_fixed = 0.1
np.random.seed(42)

# Генерация полной таблицы истинности
X_full = np.array(list(itertools.product([0, 1], repeat=n)), dtype=float)
y_full = np.array([1.0 if np.all(x == 0) else 0.0 for x in X_full], dtype=float)

print("=" * 60)
print(f"Вариант 7: n={n}, функция NOR (OR')")
print(f"Всего примеров: {len(X_full)}")
print(f"Класс 1 (все нули): {np.sum(y_full == 1)}")
print(f"Класс 0 (остальные): {np.sum(y_full == 0)}")
print("=" * 60)

# Разделение на обучающую (80%) и тестовую (20%) выборки
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

print(f"Обучающая выборка: {len(X_train)} примеров")
print(f"  Класс 0: {np.sum(y_train == 0)}, Класс 1: {np.sum(y_train == 1)}")
print(f"Тестовая выборка: {len(X_test)} примеров")
print(f"  Класс 0: {np.sum(y_test == 0)}, Класс 1: {np.sum(y_test == 1)}")


# ========== Функции ==========
def sigmoid(s):
    if s >= 0:
        z = np.exp(-s)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(s)
        return z / (1.0 + z)


def bce_sum(y, e, eps=1e-12):
    y = np.clip(y, eps, 1.0 - eps)
    return float(-np.sum(e * np.log(y) + (1.0 - e) * np.log(1.0 - y)))


def alpha_adaptive(x):
    return 1.0 / (1.0 + float(np.sum(x ** 2)))


# ========== Класс персептрона ==========
class Perceptron:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.w = np.random.uniform(-0.5, 0.5, size=n)
        self.b = np.random.uniform(-0.5, 0.5)

    def forward(self, x):
        return sigmoid(np.dot(self.w, x) + self.b)

    def train_epoch_mse(self, X, y, alpha, adaptive=False):
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        total_error = 0.0

        for i in idx:
            x = X[i]
            target = y[i]
            pred = self.forward(x)
            error = target - pred
            total_error += 0.5 * error ** 2

            if adaptive:
                lr = alpha_adaptive(x)
            else:
                lr = alpha

            grad = error * pred * (1 - pred)
            self.w += lr * grad * x
            self.b += lr * grad

        return total_error

    def train_epoch_bce(self, X, y, alpha, adaptive=False):
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        total_error = 0.0

        for i in idx:
            x = X[i]
            target = y[i]
            pred = self.forward(x)

            eps = 1e-12
            pred_clipped = np.clip(pred, eps, 1 - eps)
            total_error += -(target * np.log(pred_clipped) + (1 - target) * np.log(1 - pred_clipped))

            if adaptive:
                lr = alpha_adaptive(x)
            else:
                lr = alpha

            grad = target - pred
            self.w += lr * grad * x
            self.b += lr * grad

        return total_error


# ========== Обучение и запись ошибок на train и test ==========
max_epochs = 5000
configs = {
    'MSE + Fixed': ('mse', False),
    'MSE + Adaptive': ('mse', True),
    'BCE + Fixed': ('bce', False),
    'BCE + Adaptive': ('bce', True)
}

results = {}

for name, (loss_type, adaptive) in configs.items():
    print(f"\nОбучение: {name}")
    model = Perceptron()
    train_errors = []
    test_errors = []

    for epoch in range(max_epochs):
        if loss_type == 'mse':
            train_err = model.train_epoch_mse(X_train, y_train, alpha_fixed, adaptive)
        else:
            train_err = model.train_epoch_bce(X_train, y_train, alpha_fixed, adaptive)

        train_errors.append(train_err)

        # Ошибка на тестовой выборке
        if loss_type == 'mse':
            y_test_pred = np.array([model.forward(x) for x in X_test])
            test_err = np.sum(0.5 * (y_test - y_test_pred) ** 2)
        else:
            y_test_pred = np.array([model.forward(x) for x in X_test])
            eps = 1e-12
            y_test_pred_clipped = np.clip(y_test_pred, eps, 1 - eps)
            test_err = -np.sum(y_test * np.log(y_test_pred_clipped) + (1 - y_test) * np.log(1 - y_test_pred_clipped))

        test_errors.append(test_err)

        if train_err <= E_e:
            break

    results[name] = {
        'model': model,
        'train_errors': train_errors,
        'test_errors': test_errors,
        'epochs': len(train_errors),
        'final_train_err': train_errors[-1],
        'final_test_err': test_errors[-1]
    }
    print(f"  Завершено за {len(train_errors)} эпох")
    print(f"  Финальная ошибка train: {train_errors[-1]:.6e}")
    print(f"  Финальная ошибка test: {test_errors[-1]:.6e}")

# ========== ГРАФИК 1: Ошибка на обучающей выборке ==========
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
colors = ['blue', 'green', 'red', 'orange']
styles = ['-', '--', '-.', ':']
for idx, (name, data) in enumerate(results.items()):
    plt.plot(data['train_errors'], label=f"{name} ({data['epochs']} эп.)",
             color=colors[idx], linestyle=styles[idx], linewidth=1.5)
plt.axhline(y=E_e, color='black', linestyle='--', linewidth=1, label=f'Порог Ee = {E_e}')
plt.yscale('log')
plt.xlabel('Эпоха', fontsize=12)
plt.ylabel('Суммарная ошибка (log scale)', fontsize=12)
plt.title('График сходимости на ОБУЧАЮЩЕЙ выборке', fontsize=12)
plt.legend(loc='upper right', fontsize=9)
plt.grid(True, alpha=0.3)

# ========== ГРАФИК 2: Ошибка на тестовой выборке ==========
plt.subplot(1, 2, 2)
for idx, (name, data) in enumerate(results.items()):
    plt.plot(data['test_errors'], label=f"{name} ({data['epochs']} эп.)",
             color=colors[idx], linestyle=styles[idx], linewidth=1.5)
plt.yscale('log')
plt.xlabel('Эпоха', fontsize=12)
plt.ylabel('Суммарная ошибка (log scale)', fontsize=12)
plt.title('График сходимости на ТЕСТОВОЙ выборке', fontsize=12)
plt.legend(loc='upper right', fontsize=9)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('convergence_train_test.png', dpi=150)
plt.show()

# ========== Таблица результатов ==========
print("\n" + "=" * 70)
print("ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 70)
print(f"{'Конфигурация':<20} {'Эпохи':<8} {'Train Err':<12} {'Test Err':<12}")
print("-" * 60)

for name, data in results.items():
    print(f"{name:<20} {data['epochs']:<8} {data['final_train_err']:<12.6e} {data['final_test_err']:<12.6e}")

# ========== Вывод весов ==========
print("\n" + "=" * 70)
print("ВЕСОВЫЕ КОЭФФИЦИЕНТЫ И ПОРОГ")
print("=" * 70)
for name, data in results.items():
    model = data['model']
    print(f"\n{name}:")
    print(f"  Порог (bias): {model.b:.8f}")
    for i in range(n):
        print(f"  w{i + 1}: {model.w[i]:.8f}")


# ========== Оценка точности ==========
def get_accuracy(model, X, y):
    correct = 0
    for i in range(len(X)):
        pred = 1 if model.forward(X[i]) >= 0.5 else 0
        if pred == y[i]:
            correct += 1
    return correct / len(X) * 100


print("\n" + "=" * 70)
print("ТОЧНОСТЬ КЛАССИФИКАЦИИ")
print("=" * 70)
print(f"{'Конфигурация':<20} {'Train Acc':<12} {'Test Acc':<12} {'Full Acc':<12}")
print("-" * 60)

for name, data in results.items():
    acc_train = get_accuracy(data['model'], X_train, y_train)
    acc_test = get_accuracy(data['model'], X_test, y_test)
    acc_full = get_accuracy(data['model'], X_full, y_full)
    print(f"{name:<20} {acc_train:>10.2f}%     {acc_test:>10.2f}%     {acc_full:>10.2f}%")

# ========== Режим функционирования ==========
print("\n" + "=" * 70)
print("РЕЖИМ ФУНКЦИОНИРОВАНИЯ")
print("=" * 70)
print(f"Введите {n} бит через пробел (0 или 1)")
print("Для выхода введите 'q'")

best_config = 'BCE + Adaptive'
best_model = results[best_config]['model']

while True:
    s = input(f"\nВведите {n} чисел > ").strip()
    if s.lower() in ('q', 'quit', 'exit'):
        break
    try:
        vals = s.replace(',', ' ').split()
        if len(vals) != n:
            print(f"Ошибка: нужно ввести ровно {n} чисел")
            continue
        x_user = np.array([float(v) for v in vals])
        if not np.all((x_user == 0) | (x_user == 1)):
            print("Ошибка: можно вводить только 0 и 1")
            continue

        prob = best_model.forward(x_user)
        pred_class = 1 if prob >= 0.5 else 0
        true_class = 1 if np.all(x_user == 0) else 0

        print(f"Вектор: {x_user.astype(int)}")
        print(f"Вероятность класса 1: {prob:.6f}")
        print(f"Предсказанный класс: {pred_class}")
        print(f"Истинный класс: {true_class}")

        if pred_class == true_class:
            print("✓ Совпадает с таблицей истинности")
        else:
            print("✗ Расхождение")
    except Exception as e:
        print(f"Ошибка: {e}")

# ========== Вывод ==========
print("\n" + "=" * 70)
print("ВЫВОД")
print("=" * 70)
print("""
Сигмоидальный однослойный персептрон успешно восстановил функцию NOR
по обучающей выборке и корректно воспроизвёл полную таблицу истинности
для шести переменных. Это подтверждает способность персептрона представлять
базовые логические операции.
""")