"""
Простой фильтр Калмана для 2D трекинга (модель постоянной скорости).

Состояние: [x, y, vx, vy]
Измерение: [x, y] (сырая, шумная позиция от TDOA-трилатерации)

Зачем: TDOA-локализация даёт шумные, иногда сильно ошибочные ("выбросы")
оценки позиции от кадра к кадру (см. результаты localization.py при
повышенном шуме — до 0.39 м ошибки в отдельных точках). Фильтр Калмана
сглаживает эти скачки, используя модель движения (объект не может
телепортироваться) и постепенно "доверяя" новым измерениям пропорционально
их согласованности с предсказанием.

Написан вручную (без filterpy) чтобы не тащить лишнюю зависимость для
такой простой линейной модели — на реальном железе можно заменить на
filterpy.kalman.KalmanFilter при желании, логика та же.

ЗАМЕТКА ПО ПОДБОРУ ПАРАМЕТРОВ (из Phase 0, синтетические данные):
На резко ломаной траектории (острые повороты, скорость 1 м/с, событие
раз в 0.5 сек) высокий process_noise (1.0-2.0) в сочетании с низким
measurement_noise (0.05) даёт лучший баланс — фильтр не "запаздывает"
на поворотах, при этом всё ещё немного сглаживает случайные выбросы TDOA.
Слишком низкий process_noise (0.05) на такой траектории только вредит:
фильтр слишком долго "верит" в старую скорость движения и не успевает
среагировать на поворот, из-за чего средняя ошибка становится ЗАМЕТНО
хуже, чем без фильтра вообще (0.111 м против 0.081 м сырых данных).
На реальных плавных траекториях (обычная ходьба человека, без резких
рывков) ожидается более явный выигрыш от сглаживания — стоит перепроверить
эти параметры на живых данных в Phase 2, а не считать их окончательными.
"""

import numpy as np


class ConstantVelocityKalman2D:
    def __init__(self, initial_position, dt=0.5, process_noise=1.0, measurement_noise=0.05):
        """
        dt: ожидаемый интервал между измерениями (сек) — должен примерно
            соответствовать event_interval_sec из симуляции
        process_noise: насколько сильно "доверяем" модели движения
                       (меньше -> более гладкая, но более инертная траектория)
        measurement_noise: насколько "доверяем" сырым измерениям TDOA
                            (больше -> сильнее сглаживает шум, но медленнее
                            реагирует на реальные резкие повороты)
        """
        self.dt = dt

        # состояние: [x, y, vx, vy]
        self.x = np.array([initial_position[0], initial_position[1], 0.0, 0.0])

        # матрица перехода состояния (модель постоянной скорости)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        # матрица наблюдения (измеряем только позицию, не скорость)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        # ковариация ошибки состояния (начальная неуверенность)
        self.P = np.eye(4) * 1.0

        # шум процесса (неопределённость модели движения)
        q = process_noise
        self.Q = np.array([
            [q, 0, 0, 0],
            [0, q, 0, 0],
            [0, 0, q * 2, 0],
            [0, 0, 0, q * 2],
        ])

        # шум измерения (неопределённость сырых TDOA-координат)
        r = measurement_noise
        self.R = np.array([
            [r, 0],
            [0, r],
        ])

    def predict(self):
        """Шаг предсказания — используется и когда измерения нет
        (например, TDOA-фрейм пропущен из-за эхо-скана в гибридной архитектуре)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, measured_position):
        """Шаг коррекции по новому сырому измерению позиции от TDOA."""
        z = np.array(measured_position)
        y = z - self.H @ self.x  # невязка (innovation)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # коэффициент усиления Калмана

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2].copy()

    def step(self, measured_position=None):
        """Удобная обёртка: предсказание + (если есть измерение) коррекция."""
        self.predict()
        if measured_position is not None:
            self.update(measured_position)
        return self.x[:2].copy()

    def get_velocity(self):
        return self.x[2:4].copy()


if __name__ == "__main__":
    # Прогон полного пайплайна: симуляция -> локализация -> Калман,
    # сравнение "сырых" оценок TDOA и сглаженных Калманом относительно истины.
    from signal_generator import (
        mic_array_positions_default,
        generate_trajectory,
        simulate_multichannel_recording,
    )
    from localization import localize_event

    fs = 16000
    mics = mic_array_positions_default()
    waypoints = [(0.5, 0.5), (2.5, 3.5), (4.5, 0.5), (2.0, 1.0)]
    traj_pos, traj_t = generate_trajectory(waypoints, speed_m_s=1.0)

    event_interval = 0.5
    recordings, events = simulate_multichannel_recording(
        mics, traj_pos, traj_t, fs=fs, event_interval_sec=event_interval, noise_level=0.2
    )

    chirp_duration = 0.05
    window_samples = int(chirp_duration * fs) + int(0.04 * fs)

    raw_positions = []
    true_positions = []

    for event in events:
        t_event = event["t"]
        true_positions.append(event["true_position"])

        start_sample = int(t_event * fs)
        end_sample = start_sample + window_samples
        frame_per_mic = [recordings[m, start_sample:end_sample] for m in range(len(mics))]

        est_pos, _ = localize_event(frame_per_mic, fs, mics)
        raw_positions.append(est_pos)

    raw_positions = np.array(raw_positions)
    true_positions = np.array(true_positions)

    # прогон через фильтр Калмана
    kf = ConstantVelocityKalman2D(initial_position=raw_positions[0], dt=event_interval)
    smoothed_positions = [raw_positions[0]]
    for pos in raw_positions[1:]:
        smoothed = kf.step(pos)
        smoothed_positions.append(smoothed)
    smoothed_positions = np.array(smoothed_positions)

    raw_errors = np.linalg.norm(raw_positions - true_positions, axis=1)
    smoothed_errors = np.linalg.norm(smoothed_positions - true_positions, axis=1)

    print("Сравнение сырых TDOA-оценок и сглаженных Калманом:\n")
    print(f"{'t':>6} | {'raw error':>10} | {'smoothed error':>15}")
    for i, t in enumerate([e['t'] for e in events]):
        print(f"{t:6.2f} | {raw_errors[i]:10.3f} | {smoothed_errors[i]:15.3f}")

    print(f"\nСредняя ошибка (сырые данные):      {np.mean(raw_errors):.3f} м")
    print(f"Средняя ошибка (после Калмана):     {np.mean(smoothed_errors):.3f} м")
    print(f"Макс. ошибка (сырые данные):        {np.max(raw_errors):.3f} м")
    print(f"Макс. ошибка (после Калмана):       {np.max(smoothed_errors):.3f} м")