"""
Генератор синтетических сигналов для симуляции акустической TDOA-локализации.

Идея: у нас есть N виртуальных микрофонов с известными координатами.
Источник звука движется по заданной траектории (ground truth).
Для каждого момента времени мы генерируем короткий "чирп" (импульс),
и для каждого микрофона считаем, с какой задержкой этот импульс до него дойдёт
(расстояние / скорость звука), плюс добавляем реалистичный шум записи.

Результат: N аудио-дорожек, как будто это реально записали N микрофонов.
Дальше эти дорожки скармливаются в TDOA-алгоритм, чтобы проверить,
насколько точно он восстановит исходную (известную) траекторию.
"""

import numpy as np

SPEED_OF_SOUND = 343.0  # м/с, комнатная температура ~20°C


def generate_chirp(duration_sec, fs, f_start=800, f_end=3000):
    """
    Короткий импульс-чирп (пример звука события: шаг, щелчок, короткий стук).
    Использование чирпа (свип частоты) вместо чистого тона даёт GCC-PHAT
    более выраженный, "острый" пик корреляции — это ближе к тому, как ведут
    себя реальные бытовые звуки (шаги, хлопки), у которых широкий спектр.
    """
    t = np.linspace(0, duration_sec, int(fs * duration_sec), endpoint=False)
    freq = np.linspace(f_start, f_end, len(t))
    signal = np.sin(2 * np.pi * freq * t)
    # огибающая, чтобы не было резких щелчков на границах (влияет на GCC-PHAT)
    envelope = np.hanning(len(t))
    return signal * envelope


def mic_array_positions_default():
    """
    Расстановка 6 микрофонов в комнате 5x4 метра:
    4 угла + 2 середины длинных стен, как обсуждали в архитектуре.
    Координаты в метрах, комната: x in [0,5], y in [0,4].
    """
    return np.array([
        [0.0, 0.0],   # угол 1
        [5.0, 0.0],   # угол 2
        [5.0, 4.0],   # угол 3
        [0.0, 4.0],   # угол 4
        [0.0, 2.0],   # середина левой стены
        [5.0, 2.0],   # середина правой стены
    ])


def generate_trajectory(waypoints, speed_m_s=1.0, fs_traj=20):
    """
    Строит плавную траекторию источника звука по опорным точкам (waypoints),
    двигаясь с постоянной скоростью между ними.

    waypoints: список (x, y) точек, через которые проходит источник
    speed_m_s: скорость перемещения источника (например, скорость шага человека)
    fs_traj: частота дискретизации траектории (точек в секунду)

    Возвращает: (positions, timestamps)
    """
    positions = []
    timestamps = []
    t_current = 0.0

    for i in range(len(waypoints) - 1):
        p0 = np.array(waypoints[i], dtype=float)
        p1 = np.array(waypoints[i + 1], dtype=float)
        segment_len = np.linalg.norm(p1 - p0)
        segment_duration = segment_len / speed_m_s
        n_points = max(int(segment_duration * fs_traj), 2)

        for k in range(n_points):
            alpha = k / n_points
            pos = p0 + alpha * (p1 - p0)
            positions.append(pos)
            timestamps.append(t_current + k / fs_traj)

        t_current += segment_duration

    return np.array(positions), np.array(timestamps)


def simulate_multichannel_recording(
    mic_positions,
    trajectory_positions,
    trajectory_timestamps,
    fs=16000,
    event_interval_sec=0.5,
    chirp_duration=0.05,
    noise_level=0.02,
    seed=42,
):
    """
    Основная функция симуляции.

    По ходу движения источника (заданного траекторией) через равные интервалы
    времени (event_interval_sec) "испускается" звуковое событие (шаг/щелчок).
    Для каждого события и для каждого микрофона считается задержка
    распространения звука (расстояние / скорость звука) и в общий сигнал
    микрофона добавляется чирп со сдвигом на эту задержку.

    Возвращает:
        recordings: массив shape (n_mics, n_samples) — многоканальная запись
        event_log: список словарей с "правдой" о каждом событии
                   (истинное время, истинная позиция источника) — для сверки
                   с тем, что потом восстановит алгоритм локализации
    """
    rng = np.random.default_rng(seed)
    n_mics = len(mic_positions)

    total_duration = trajectory_timestamps[-1] + 1.0  # запас в конце
    n_samples = int(total_duration * fs)
    recordings = np.zeros((n_mics, n_samples))

    chirp = generate_chirp(chirp_duration, fs)
    event_log = []

    event_times = np.arange(0, trajectory_timestamps[-1], event_interval_sec)

    for t_event in event_times:
        # находим позицию источника в момент t_event (линейная интерполяция по траектории)
        source_pos = _interpolate_position(trajectory_positions, trajectory_timestamps, t_event)

        event_log.append({"t": t_event, "true_position": source_pos.copy()})

        for m_idx, mic_pos in enumerate(mic_positions):
            distance = np.linalg.norm(source_pos - mic_pos)
            delay_sec = distance / SPEED_OF_SOUND
            start_sample = int((t_event + delay_sec) * fs)
            end_sample = start_sample + len(chirp)

            if end_sample < n_samples:
                # затухание по расстоянию (простая модель 1/r, r>=0.5 чтобы не делить на 0)
                attenuation = 1.0 / max(distance, 0.5)
                recordings[m_idx, start_sample:end_sample] += chirp * attenuation

    # добавляем шум записи на все каналы (имитация реального железа)
    recordings += rng.normal(0, noise_level, recordings.shape)

    return recordings, event_log


def _interpolate_position(positions, timestamps, t_query):
    """Линейная интерполяция позиции источника в произвольный момент времени."""
    if t_query <= timestamps[0]:
        return positions[0]
    if t_query >= timestamps[-1]:
        return positions[-1]

    idx = np.searchsorted(timestamps, t_query) - 1
    idx = max(0, min(idx, len(timestamps) - 2))

    t0, t1 = timestamps[idx], timestamps[idx + 1]
    p0, p1 = positions[idx], positions[idx + 1]
    alpha = (t_query - t0) / (t1 - t0) if t1 > t0 else 0
    return p0 + alpha * (p1 - p0)


if __name__ == "__main__":
    # Быстрая самопроверка модуля
    mics = mic_array_positions_default()
    waypoints = [(0.5, 0.5), (2.5, 3.5), (4.5, 0.5)]
    traj_pos, traj_t = generate_trajectory(waypoints, speed_m_s=1.0)

    print(f"Микрофоны:\n{mics}")
    print(f"Точек траектории: {len(traj_pos)}, длительность: {traj_t[-1]:.2f} сек")

    recordings, events = simulate_multichannel_recording(mics, traj_pos, traj_t)
    print(f"Записи: shape={recordings.shape}")
    print(f"Событий сгенерировано: {len(events)}")
    print(f"Первое событие: t={events[0]['t']:.2f}, позиция={events[0]['true_position']}")
    print(f"Последнее событие: t={events[-1]['t']:.2f}, позиция={events[-1]['true_position']}")