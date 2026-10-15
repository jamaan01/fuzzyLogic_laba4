import numpy as np

SMALL_NUMBER = 1e-12

def frobenius_distances_squared(matrix, centers):
    differences = centers - matrix
    return np.sum(differences * differences, axis=(1, 2))


def all_frobenius_distances_squared(data, centers):
    differences = data[:, np.newaxis, :, :] - centers[np.newaxis, :, :, :]
    return np.sum(differences * differences, axis=(2, 3))


def prepare_matrices(data):
    centered_data = data - np.mean(data, axis=0, keepdims=True)
    norms = np.sqrt(
        np.sum(centered_data * centered_data, axis=(1, 2), keepdims=True)
    )
    norms = np.maximum(norms, SMALL_NUMBER)
    return centered_data / norms


def update_fcm_memberships(distances_squared, fuzziness):
    object_count, cluster_count = distances_squared.shape
    memberships = np.zeros((object_count, cluster_count), dtype=float)
    power = 1.0 / (fuzziness - 1.0)

    for object_index in range(object_count):
        row = distances_squared[object_index]
        zero_indices = np.flatnonzero(row <= SMALL_NUMBER)

        if zero_indices.size:
            memberships[object_index, zero_indices] = 1.0 / zero_indices.size
        else:
            inverse_distances = row ** (-power)
            memberships[object_index] = inverse_distances / np.sum(inverse_distances)

    return memberships


def calculate_centers(data, weights):
    numerator = np.einsum("nc,nij->cij", weights, data)
    denominator = np.sum(weights, axis=0)[:, np.newaxis, np.newaxis]
    denominator = np.maximum(denominator, SMALL_NUMBER)
    return numerator / denominator


def matrix_fcm(data, cluster_count, fuzziness=2.0, epsilon=1e-5,
               max_iterations=300, seed=9):
    if fuzziness <= 1:
        raise ValueError("Параметр нечіткості повинен бути більшим за 1.")
    if cluster_count < 2 or cluster_count > len(data):
        raise ValueError("Некоректна кількість кластерів.")

    random_generator = np.random.default_rng(seed)
    memberships = random_generator.random((len(data), cluster_count))
    memberships /= np.sum(memberships, axis=1, keepdims=True)
    objective_history = []

    for iteration in range(1, max_iterations + 1):
        old_memberships = memberships.copy()
        weights = memberships ** fuzziness
        centers = calculate_centers(data, weights)
        distances_squared = all_frobenius_distances_squared(data, centers)
        memberships = update_fcm_memberships(distances_squared, fuzziness)

        objective = np.sum((memberships ** fuzziness) * distances_squared)
        objective_history.append(float(objective))

        difference = np.max(np.abs(memberships - old_memberships))
        if difference < epsilon:
            break

    labels = np.argmax(memberships, axis=1)
    return centers, memberships, labels, objective_history, iteration


def calculate_pcm_scales(data, centers, memberships, fuzziness):
    distances_squared = all_frobenius_distances_squared(data, centers)
    weights = memberships ** fuzziness
    numerator = np.sum(weights * distances_squared, axis=0)
    denominator = np.maximum(np.sum(weights, axis=0), SMALL_NUMBER)
    scales = numerator / denominator
    return np.maximum(scales, 1e-6)


def pcm_typicalities(distances_squared, scales, fuzziness):
    power = 1.0 / (fuzziness - 1.0)
    ratios = distances_squared / scales[np.newaxis, :]
    return 1.0 / (1.0 + ratios ** power)


def adaptive_matrix_pcm(data, initial_centers, initial_memberships,
                        fuzziness=2.0, learning_rate=0.35,
                        learning_decay=0.02, seed=9):
    centers = initial_centers.copy()
    scales = calculate_pcm_scales(
        data, centers, initial_memberships, fuzziness
    )

    random_generator = np.random.default_rng(seed)
    processing_order = random_generator.permutation(len(data))

    for step, object_index in enumerate(processing_order):
        matrix = data[object_index]
        distances_squared = frobenius_distances_squared(matrix, centers)
        typicalities = pcm_typicalities(
            distances_squared[np.newaxis, :], scales, fuzziness
        )[0]
        current_rate = learning_rate / (1.0 + learning_decay * step)

        for cluster_index in range(len(centers)):
            weight = typicalities[cluster_index] ** fuzziness
            centers[cluster_index] += (
                current_rate * weight * (matrix - centers[cluster_index])
            )

    final_distances = all_frobenius_distances_squared(data, centers)
    typicalities = pcm_typicalities(final_distances, scales, fuzziness)
    labels = np.argmax(typicalities, axis=1)
    return centers, typicalities, labels, scales


def create_demo_data(seed=9):
    random_generator = np.random.default_rng(seed)
    prototypes = np.array(
        [
            [[1.0, 0.1], [0.1, 0.0]],
            [[0.0, 0.1], [0.1, 1.0]],
            [[0.1, 1.0], [1.0, 0.1]],
        ],
        dtype=float,
    )

    objects = []
    for prototype in prototypes:
        for _ in range(10):
            noise = random_generator.normal(0.0, 0.07, size=(2, 2))
            objects.append(prototype + noise)

    objects.append(np.array([[0.55, 0.55], [0.55, 0.55]]))
    objects.append(np.array([[0.85, -0.65], [-0.60, 0.75]]))
    return prepare_matrices(np.array(objects, dtype=float))


def read_number(prompt, default, number_type=float):
    value = input(prompt).strip()
    if not value:
        return default
    return number_type(value)


def print_centers(centers):
    for index, center in enumerate(centers, start=1):
        print(f"Центр {index}:")
        print(np.array2string(center, precision=4, suppress_small=True))


def main():
    try:
        cluster_count = read_number("Кількість кластерів [3]: ", 3, int)
        fuzziness = read_number("Параметр нечіткості beta [2]: ", 2.0)
        epsilon = read_number("Точність epsilon [0.00001]: ", 1e-5)
        learning_rate = read_number("Крок навчання eta [0.35]: ", 0.35)
    except ValueError:
        print("Помилка: параметри повинні бути числами.")
        return

    data = create_demo_data(seed=9)

    fcm_result = matrix_fcm(
        data,
        cluster_count=cluster_count,
        fuzziness=fuzziness,
        epsilon=epsilon,
        seed=9,
    )
    fcm_centers, memberships, fcm_labels, _history, iterations = fcm_result

    pcm_result = adaptive_matrix_pcm(
        data,
        initial_centers=fcm_centers,
        initial_memberships=memberships,
        fuzziness=fuzziness,
        learning_rate=learning_rate,
        seed=9,
    )
    pcm_centers, typicalities, pcm_labels, _scales = pcm_result

    print("\nMatrix FCM")
    print(f"Кількість ітерацій: {iterations}")
    print_centers(fcm_centers)
    print("Кластери об'єктів:")
    print(" ".join(str(label + 1) for label in fcm_labels))

    print("\nAdaptive Matrix PCM")
    print_centers(pcm_centers)
    print("Кластери об'єктів:")
    print(" ".join(str(label + 1) for label in pcm_labels))

    anomaly_numbers = [
        str(index + 1)
        for index, row in enumerate(typicalities)
        if np.max(row) < 0.35
    ]
    print("Аномальні об'єкти:")
    print(" ".join(anomaly_numbers) if anomaly_numbers else "немає")


if __name__ == "__main__":
    main()
